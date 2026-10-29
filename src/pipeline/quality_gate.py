"""Quality gate: stop a retrained model from being published if it is not good enough.

The weekly pipeline retrains everything. Before the new models go live they must still
 - beat the popularity baseline in retrieval, and
 - beat plain retrieval order once the ranker re-orders the candidates.
Exit code 1 means "do not publish", which makes the Airflow task (and the run) fail.

Run:
    python src/pipeline/quality_gate.py --retrieval-metrics outputs/retrieval/metrics.json \
        --ranker-metrics outputs/ranking_v3/candidate_ranker_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

RETRIEVAL_KEYS = ["recall@10", "recall@100", "ndcg@10"]
RANKER_KEYS = ["ndcg@10", "recall@10"]
MIN_RANKER_AUC = 0.60        # sanity floor, not a target


def check(retrieval, ranker):
    """Return a list of human-readable failures (empty list = good to publish)."""
    failures = []
    model, baseline = retrieval["two_tower"], retrieval["popularity"]
    for key in RETRIEVAL_KEYS:
        if model[key] <= baseline[key]:
            failures.append(f"two-tower {key} {model[key]} does not beat popularity {baseline[key]}")

    reranked, plain = ranker["ranker_order"], ranker["retrieval_order"]
    for key in RANKER_KEYS:
        if reranked[key] <= plain[key]:
            failures.append(f"ranker {key} {reranked[key]} does not beat retrieval order {plain[key]}")

    if ranker["out_of_fold_auc"] < MIN_RANKER_AUC:
        failures.append(f"ranker out-of-fold AUC {ranker['out_of_fold_auc']} is below {MIN_RANKER_AUC}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-metrics", type=Path, required=True)
    ap.add_argument("--ranker-metrics", type=Path, required=True)
    args = ap.parse_args()

    failures = check(json.loads(args.retrieval_metrics.read_text()),
                     json.loads(args.ranker_metrics.read_text()))
    if failures:
        print("QUALITY GATE FAILED:")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    main()
