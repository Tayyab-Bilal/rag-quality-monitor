"""Hook 3: run the pipeline over the golden set and score it."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from ..judge.base import Embedder, Judge
from ..judge.llm import PROMPT_VERSION
from ..metrics import generation, retrieval
from ..models import Chunk, GoldenCase, RetrievalResult

Retriever = Callable[[str, str, int], Awaitable[RetrievalResult]]
Generator = Callable[[str, list[Chunk]], Awaitable[str]]


class CaseResult(BaseModel):
    case_id: str
    route: str
    retrieved_ids: list[str]
    algorithmic: dict[str, float]
    judged: dict[str, float] = {}
    unsupported_claims: list[str] = []
    answer: str | None = None


class RouteSummary(BaseModel):
    cases: int
    algorithmic: dict[str, float]
    judged: dict[str, float]


class SuiteReport(BaseModel):
    k: int
    config_hash: str
    cases: list[CaseResult]
    routes: dict[str, RouteSummary]


def _means(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: statistics.fmean(row[key] for row in rows if key in row) for key in keys}


async def run_suite(
    golden: list[GoldenCase],
    retriever: Retriever,
    generator: Generator | None = None,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
    *,
    k: int = 5,
    tenant_id: str = "suite",
) -> SuiteReport:
    """Algorithmic retrieval metrics always; judged generation metrics when a generator and judge exist."""
    results: list[CaseResult] = []
    for case in golden:
        found = await retriever(tenant_id, case.question, k)
        ids = [c.id for c in found.chunks]
        rel = case.relevant_chunk_ids
        result = CaseResult(
            case_id=case.case_id,
            route=case.route,
            retrieved_ids=ids,
            algorithmic={
                "recall_at_k": retrieval.recall_at_k(ids, rel, k),
                "precision_at_k": retrieval.precision_at_k(ids, rel, k),
                "hit_rate": retrieval.hit_rate(ids, rel, k),
                "mrr": retrieval.mrr(ids, rel, k),
                "ndcg_at_k": retrieval.ndcg_at_k(ids, rel, k),
            },
        )
        if generator and judge:
            result.answer = await generator(case.question, found.chunks)
            faith = await generation.faithfulness(result.answer, found.chunks, judge)
            judged: dict[str, float | None] = {
                "faithfulness": faith.score,
                "context_precision": await generation.context_precision(case.question, found.chunks, judge),
            }
            if embedder:
                judged["answer_relevance"] = await generation.answer_relevance(
                    case.question, result.answer, judge, embedder
                )
            if case.reference_answer:
                judged["context_recall"] = await generation.context_recall(case.reference_answer, found.chunks, judge)
            result.judged = {name: v for name, v in judged.items() if v is not None}
            result.unsupported_claims = faith.unsupported_claims
        results.append(result)

    by_route: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_route[r.route].append(r)
    routes = {
        route: RouteSummary(
            cases=len(rows), algorithmic=_means([r.algorithmic for r in rows]), judged=_means([r.judged for r in rows])
        )
        for route, rows in sorted(by_route.items())
    }
    blob = json.dumps(
        {
            "judge": judge.model if judge else None,
            "embedder": getattr(embedder, "model", type(embedder).__name__) if embedder else None,
            "prompt": PROMPT_VERSION,
            "k": k,
            "golden": hashlib.sha256(json.dumps([c.model_dump() for c in golden], sort_keys=True).encode()).hexdigest(),
        },
        sort_keys=True,
    )
    return SuiteReport(k=k, config_hash=hashlib.sha256(blob.encode()).hexdigest()[:12], cases=results, routes=routes)
