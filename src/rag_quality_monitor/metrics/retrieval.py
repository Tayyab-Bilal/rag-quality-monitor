"""Dimension 1, algorithmic half: retrieval metrics against labelled chunk ids.

Pure and deterministic, so they can gate a merge. Conventions (all tested):
- Duplicate ids in `retrieved` count once, at their first position.
- Only the first k distinct ids are looked at; if fewer come back, precision still divides by k,
  so a retriever that returns too little is penalised.
- An empty `relevant` set is a labelling bug, so it raises instead of returning a flattering 0 or 1.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    seen: dict[str, None] = dict.fromkeys(retrieved)
    return list(seen)[:k]


def _check(relevant: Collection[str]) -> set[str]:
    if not relevant:
        raise ValueError("no relevant ids: the case is not labelled")
    return set(relevant)


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the relevant ids found in the top k."""
    rel = _check(relevant)
    return len(rel & set(_top_k(retrieved, k))) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the k slots that hold a relevant id."""
    rel = _check(relevant)
    return len(rel & set(_top_k(retrieved, k))) / k


def hit_rate(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any relevant id is in the top k, else 0.0. The mean over cases is the hit rate."""
    rel = _check(relevant)
    return 1.0 if rel & set(_top_k(retrieved, k)) else 0.0


def mrr(retrieved: Sequence[str], relevant: Collection[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none). `k` optionally cuts the list."""
    rel = _check(relevant)
    ids = _top_k(retrieved, k if k is not None else max(len(retrieved), 1))
    for rank, chunk_id in enumerate(ids, start=1):
        if chunk_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance nDCG with a log2 discount: DCG / ideal DCG."""
    rel = _check(relevant)
    ids = _top_k(retrieved, k)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(ids, start=1) if cid in rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
    return dcg / ideal
