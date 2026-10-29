"""Fast tests for the logic that is easy to get subtly wrong. No dataset or trained model needed.

Run:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

import ab_test
import prepare_data
import publish
import quality_gate
import train_ranker


# ---------------------------------------------------------------- data preparation

def test_time_split_never_uses_the_future():
    ratings, movies = prepare_data.make_synthetic(n_users=300, n_items=100, n_rows=20_000)
    df = prepare_data.clean(ratings, movies)
    train, val, test, *_ = prepare_data.temporal_split(df)
    assert train["ts"].max() <= val["ts"].min()
    assert val["ts"].max() <= test["ts"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_cleaning_drops_users_with_too_few_ratings_and_labels_likes():
    rows = [(u, i, 4.0, u * 10 + i) for u in range(1, 7) for i in range(1, 7)]
    rows.append((99, 1, 5.0, 999))                      # one rating only: must be dropped
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "ts"])
    movies = pd.DataFrame({"item_id": range(1, 7), "title": list("abcdef"), "genres": ["Drama"] * 6})
    out = prepare_data.clean(ratings, movies)
    assert 99 not in set(out["user_id"])
    assert len(out) == 36
    assert set(out["label"]) == {1}                     # rating 4.0 counts as a like


# ---------------------------------------------------------------- quality gate

RETRIEVAL = {"two_tower": {"recall@10": 0.0468, "recall@100": 0.2372, "ndcg@10": 0.1099},
             "popularity": {"recall@10": 0.0321, "recall@100": 0.1501, "ndcg@10": 0.0869}}


def ranker_metrics(ranker_ndcg, ranker_recall, auc=0.70):
    return {"out_of_fold_auc": auc,
            "retrieval_order": {"ndcg@10": 0.1102, "recall@10": 0.0473},
            "ranker_order": {"ndcg@10": ranker_ndcg, "recall@10": ranker_recall}}


def test_quality_gate_passes_a_better_ranker():
    assert quality_gate.check(RETRIEVAL, ranker_metrics(0.1307, 0.0574)) == []


def test_quality_gate_blocks_a_worse_ranker():
    failures = quality_gate.check(RETRIEVAL, ranker_metrics(0.0691, 0.0284))
    assert len(failures) == 2


def test_quality_gate_blocks_retrieval_that_loses_to_popularity():
    weak = {"two_tower": {"recall@10": 0.02, "recall@100": 0.10, "ndcg@10": 0.05},
            "popularity": RETRIEVAL["popularity"]}
    assert len(quality_gate.check(weak, ranker_metrics(0.1307, 0.0574))) == 3


# ---------------------------------------------------------------- publish

def test_publish_swaps_repository_and_keeps_previous(tmp_path):
    new, live = tmp_path / "new", tmp_path / "live" / "model_repository"
    (new / "m" / "1").mkdir(parents=True)
    (new / "m" / "1" / "model.onnx").write_text("v1")
    publish.swap_repository(new, live)
    (new / "m" / "1" / "model.onnx").write_text("v2")
    publish.swap_repository(new, live)
    assert (live / "m" / "1" / "model.onnx").read_text() == "v2"
    assert (live.with_name("model_repository.old") / "m" / "1" / "model.onnx").read_text() == "v1"


# ---------------------------------------------------------------- A/B test maths

def test_arm_assignment_is_stable_and_balanced():
    arms = [ab_test.assign_arm(u, "salt") for u in range(5000)]
    assert arms == [ab_test.assign_arm(u, "salt") for u in range(5000)]       # same user, same arm
    assert 0.45 < arms.count("A") / len(arms) < 0.55
    assert arms != [ab_test.assign_arm(u, "other-salt") for u in range(5000)]  # salt reshuffles


def test_user_metrics_on_a_hand_checked_example():
    m = ab_test.user_metrics([10, 20, 30], {20}, k=3)      # the liked movie is ranked second
    assert m["hit"] == 1.0
    assert m["ctr"] == pytest.approx(1 / 3)
    assert m["recall"] == 1.0
    assert m["ndcg"] == pytest.approx(1 / np.log2(3))      # discount at rank 2, ideal is rank 1


def test_ranking_metrics_reward_putting_the_hit_first():
    first = train_ranker.rank_metrics([np.array([6, 5, 7])], [{6}], ks=(3,))
    second = train_ranker.rank_metrics([np.array([5, 6, 7])], [{6}], ks=(3,))
    assert first["ndcg@3"] == 1.0
    assert second["ndcg@3"] == pytest.approx(1 / np.log2(3), abs=1e-4)
    assert first["recall@3"] == second["recall@3"] == 1.0
