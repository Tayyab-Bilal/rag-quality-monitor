"""Shared data models. Everything that crosses a module boundary is defined here."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A RAW retrieved chunk. The id is required: without it nothing can be traced back."""

    id: str = Field(min_length=1)
    doc_id: str
    text: str
    score: float


class RetrievalResult(BaseModel):
    query: str
    chunks: list[Chunk]
    k: int
    latency_ms: float


class Sample(BaseModel):
    sample_id: str
    tenant_id: str
    route: str
    query: str
    chunks: list[Chunk]
    answer: str
    created_at: float


class GoldenCase(BaseModel):
    case_id: str
    route: str
    question: str
    relevant_chunk_ids: list[str] = Field(min_length=1)
    reference_answer: str | None = None


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence_chunk_id: str | None = None


class Alert(BaseModel):
    kind: str
    tenant_id: str
    route: str
    message: str
    value: float | None = None
    threshold: float | None = None


class BatchRecord(BaseModel):
    """Aggregates of one Hook 2 batch. Kept so later batches can be compared with it."""

    batch_id: str
    tenant_id: str
    route: str
    created_at: float
    config_hash: str
    scored: int
    failed: int
    faithfulness_mean: float | None = None
    faithfulness_min: float | None = None
    answer_relevance_mean: float | None = None
    context_precision_mean: float | None = None
    judge_calls: int = 0
    drift: list[Alert] = Field(default_factory=list)
