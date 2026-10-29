"""Weekly retraining pipeline for the recommender.

    prepare_data -> build_features -> train_retrieval -> build_faiss -> train_ranker
        -> quality_gate -> export_triton -> publish

Every task is a plain command line call to a script in this repo, so each step can also be run
by hand. The quality gate fails the run if the new models are worse than the baselines, and then
nothing is published: serving keeps using the previous models.

Set RECSYS_ROOT to the folder that holds `src/` (default /opt/recsys). Artifacts go to
$RECSYS_ROOT/outputs, and the live model repository is $RECSYS_ROOT/live/model_repository.
Written for Airflow 2.x and 3.x.
"""
import os
from datetime import datetime, timedelta

from airflow import DAG

try:                                                    # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:                                     # Airflow 2
    from airflow.operators.bash import BashOperator

ROOT = os.environ.get("RECSYS_ROOT", "/opt/recsys")
OUT = f"{ROOT}/outputs"
REDIS = os.environ.get("RECSYS_REDIS", "localhost:6379")
PY = f"cd {ROOT} && python"

default_args = {
    "owner": "recsys",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="recsys_weekly_retrain",
    description="Retrain retrieval + ranker, check quality, publish to Triton and clear the cache",
    default_args=default_args,
    start_date=datetime(2026, 10, 25),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["recsys"],
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"{PY} src/data/prepare_data.py --raw {ROOT}/data/raw/ml-25m --out {OUT}/processed",
    )
    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"{PY} src/features/build_features.py --data {OUT}/processed --out {OUT}/features",
    )
    train_retrieval = BashOperator(
        task_id="train_retrieval",
        bash_command=f"{PY} src/retrieval/train_two_tower.py --data {OUT}/processed "
                     f"--out {OUT}/retrieval --epochs 3 --batch 4096",
        execution_timeout=timedelta(hours=4),
    )
    build_faiss = BashOperator(
        task_id="build_faiss",
        bash_command=f"{PY} src/retrieval/build_faiss_index.py --emb {OUT}/retrieval --out {OUT}/faiss",
    )
    train_ranker = BashOperator(
        task_id="train_ranker",
        bash_command=f"{PY} src/ranking/train_candidate_ranker.py --data {OUT}/processed "
                     f"--features {OUT}/features --retrieval {OUT}/retrieval --faiss {OUT}/faiss "
                     f"--out {OUT}/ranking_v3",
    )
    quality_gate = BashOperator(
        task_id="quality_gate",
        bash_command=f"{PY} src/pipeline/quality_gate.py "
                     f"--retrieval-metrics {OUT}/retrieval/metrics.json "
                     f"--ranker-metrics {OUT}/ranking_v3/candidate_ranker_metrics.json",
        retries=0,                                      # a failed gate is a result, not a glitch
    )
    export_triton = BashOperator(
        task_id="export_triton",
        bash_command=f"{PY} src/serving/export_triton.py --artifacts-root {OUT} --data processed "
                     f"--retrieval retrieval --ranking ranking_v3 --out model_repository",
    )
    publish = BashOperator(
        task_id="publish",
        bash_command=f"{PY} src/pipeline/publish.py --src {OUT}/model_repository "
                     f"--live {ROOT}/live/model_repository --redis {REDIS}",
    )

    (prepare_data >> build_features >> train_retrieval >> build_faiss
     >> train_ranker >> quality_gate >> export_triton >> publish)
