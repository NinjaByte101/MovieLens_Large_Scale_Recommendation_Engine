"""rank_metrics edge cases that test_core.py does not cover."""
import math

import pytest

from train_ranker import rank_metrics


def test_rank_metrics_perfect_and_empty():
    perfect = rank_metrics([[7, 1, 2]], [{7}], ks=(10,))
    assert perfect == {"recall@10": 1.0, "ndcg@10": 1.0}
    miss = rank_metrics([[1, 2, 3]], [{9}], ks=(10,))
    assert miss == {"recall@10": 0.0, "ndcg@10": 0.0}


def test_rank_metrics_relevant_movie_at_rank_two():
    result = rank_metrics([[1, 7, 2]], [{7}], ks=(10,))
    assert result["recall@10"] == 1.0
    assert result["ndcg@10"] == pytest.approx(1 / math.log2(3), abs=1e-4)


def test_rank_metrics_short_candidate_list_does_not_crash():
    # fewer candidates than K used to break the discount broadcast
    result = rank_metrics([[1, 2]], [{2}], ks=(10, 50))
    assert result["recall@50"] == 1.0
