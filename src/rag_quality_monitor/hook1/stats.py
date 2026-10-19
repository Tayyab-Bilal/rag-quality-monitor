"""Hook 1, per query: statistics from what the retriever already returned. No LLM, near-zero cost."""

from __future__ import annotations

import statistics

from pydantic import BaseModel

from ..models import RetrievalResult


class RetrievalStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    score_gap: float
    count: int
    k: int
    empty: bool
    latency_ms: float


def retrieval_stats(result: RetrievalResult) -> RetrievalStats:
    """Score distribution and the gap between the top score and the k-th (or last) score.

    The gap is the early warning: when every chunk scores about the same, the right chunk can still be
    in the top k (judged precision looks fine) while the retriever has lost its ability to discriminate.
    """
    scores = sorted((c.score for c in result.chunks), reverse=True)
    if not scores:
        return RetrievalStats(
            mean=0.0, std=0.0, min=0.0, max=0.0, score_gap=0.0, count=0, k=result.k, empty=True,
            latency_ms=result.latency_ms,
        )  # fmt: skip
    return RetrievalStats(
        mean=statistics.fmean(scores),
        std=statistics.pstdev(scores),
        min=scores[-1],
        max=scores[0],
        score_gap=scores[0] - scores[min(result.k, len(scores)) - 1],
        count=len(scores),
        k=result.k,
        empty=False,
        latency_ms=result.latency_ms,
    )
