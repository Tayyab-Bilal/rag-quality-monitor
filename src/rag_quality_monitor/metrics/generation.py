"""Dimension 2 (plus judged context metrics): reference-free scores that use a judge.

Per-claim verdicts are returned, not just a number: the unsupported claims are the diagnosis.
"""

from __future__ import annotations

import asyncio
import math

from pydantic import BaseModel

from ..judge.base import Embedder, Judge
from ..models import Chunk, ClaimVerdict


class FaithfulnessResult(BaseModel):
    score: float | None
    verdicts: list[ClaimVerdict]
    unsupported_claims: list[str]
    reason: str | None = None


async def faithfulness(answer: str, chunks: list[Chunk], judge: Judge) -> FaithfulnessResult:
    """Supported claims / all claims. An answer with no claims has no score, not a perfect one."""
    claims = await judge.extract_claims(answer)
    if not claims:
        return FaithfulnessResult(score=None, verdicts=[], unsupported_claims=[], reason="answer has no claims")
    verdicts = await judge.verify_claims(claims, chunks)
    unsupported = [v.claim for v in verdicts if not v.supported]
    return FaithfulnessResult(
        score=(len(verdicts) - len(unsupported)) / len(verdicts), verdicts=verdicts, unsupported_claims=unsupported
    )


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


async def answer_relevance(
    question: str, answer: str, judge: Judge, embedder: Embedder, n: int = 3
) -> float | None:
    """Mean cosine between the real question and questions the judge says the answer would satisfy.

    None when the judge produced no questions (nothing to measure, like an answer with no claims). A 0.0
    would be indistinguishable from "answers a different question" and would drag aggregates down."""
    generated = await judge.generate_questions(answer, n)
    if not generated:
        return None
    vectors = await embedder.embed([question, *generated])
    return sum(_cosine(vectors[0], v) for v in vectors[1:]) / len(generated)


async def context_precision(question: str, chunks: list[Chunk], judge: Judge) -> float:
    """Rank-aware average precision over judged relevance. A relevant chunk at rank 1 beats rank 5."""
    flags = await asyncio.gather(*(judge.chunk_relevant(question, c) for c in chunks))
    hits, total = 0, 0.0
    for rank, relevant in enumerate(flags, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


async def context_recall(reference_answer: str, chunks: list[Chunk], judge: Judge) -> float | None:
    """Share of the reference answer's claims that some chunk supports. Needs a reference."""
    claims = await judge.extract_claims(reference_answer)
    if not claims:
        return None
    verdicts = await judge.verify_claims(claims, chunks)
    return sum(v.supported for v in verdicts) / len(verdicts)
