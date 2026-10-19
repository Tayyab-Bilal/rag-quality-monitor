import math

import pytest

from rag_quality_monitor.metrics.retrieval import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k

R = ["a", "b", "c", "d", "e"]


def test_recall_known_values():
    assert recall_at_k(R, {"b", "z"}, 5) == 0.5
    assert recall_at_k(R, {"b", "z"}, 1) == 0.0
    assert recall_at_k(R, {"a", "b"}, 2) == 1.0


def test_precision_known_values():
    assert precision_at_k(R, {"a", "c"}, 4) == 0.5
    assert precision_at_k(R, {"a"}, 1) == 1.0


def test_precision_divides_by_k_when_fewer_results_come_back():
    assert precision_at_k(["a"], {"a"}, 5) == 0.2


def test_recall_and_hit_rate_with_k_larger_than_results():
    assert recall_at_k(["a"], {"a", "b"}, 10) == 0.5
    assert hit_rate(["a"], {"a"}, 10) == 1.0


def test_hit_rate_is_binary():
    assert hit_rate(R, {"c"}, 3) == 1.0
    assert hit_rate(R, {"c"}, 2) == 0.0


def test_mrr_uses_rank_of_first_relevant():
    assert mrr(R, {"c", "e"}) == pytest.approx(1 / 3)
    assert mrr(R, {"z"}) == 0.0
    assert mrr(R, {"c"}, k=2) == 0.0


def test_ndcg_known_values():
    assert ndcg_at_k(["a", "x"], {"a"}, 2) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    # two relevant, one found at rank 2: dcg = 1/log2(3), ideal = 1 + 1/log2(3)
    assert ndcg_at_k(["x", "a", "y"], {"a", "b"}, 3) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_ndcg_prefers_relevant_chunk_first():
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) > ndcg_at_k(["x", "y", "a"], {"a"}, 3)


def test_duplicates_count_once():
    assert recall_at_k(["a", "a", "a", "b"], {"a", "b"}, 2) == 1.0
    assert precision_at_k(["a", "a", "a"], {"a"}, 3) == pytest.approx(1 / 3)
    assert mrr(["x", "x", "a"], {"a"}) == 0.5


@pytest.mark.parametrize("fn", [recall_at_k, precision_at_k, hit_rate, ndcg_at_k])
def test_no_relevant_ids_is_an_error_not_a_score(fn):
    with pytest.raises(ValueError):
        fn(R, set(), 3)


def test_non_positive_k_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k(R, {"a"}, 0)
