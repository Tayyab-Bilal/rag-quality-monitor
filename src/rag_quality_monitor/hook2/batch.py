"""Hook 2, step 3: batch-score pending samples with the reference-free metrics."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid

from pydantic import BaseModel

from ..alerts import AlertSink
from ..judge.base import CountingJudge, Embedder, Judge
from ..judge.llm import PROMPT_VERSION
from ..metrics.generation import answer_relevance, context_precision, faithfulness
from ..models import Alert, BatchRecord
from ..store import Store
from .drift import BASELINE_BATCHES, check_drift

log = logging.getLogger(__name__)

METRIC_SET = ["faithfulness", "answer_relevance", "context_precision"]
ATTRIBUTION_THRESHOLD = 0.80


class BatchResult(BaseModel):
    skipped: bool
    reason: str | None = None
    batch: BatchRecord | None = None
    left_pending: int = 0
    alerts: list[Alert] = []


def config_hash(judge: Judge, embedder: Embedder | None = None) -> str:
    """Judge model + embedder + prompt version + metric set. Results are only comparable when this matches
    (answer relevance depends on the embedder as much as on the judge)."""
    embedder_id = None if embedder is None else getattr(embedder, "model", type(embedder).__name__)
    blob = json.dumps(
        {"judge": judge.model, "embedder": embedder_id, "prompt": PROMPT_VERSION, "metrics": METRIC_SET}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


async def run_batch(
    store: Store,
    judge: Judge,
    embedder: Embedder,
    tenant_id: str,
    route: str,
    *,
    min_samples: int = 10,
    max_age: float = 3600.0,
    lease: float = 600.0,
    max_judge_calls: int = 500,
    limit: int = 50,
    sink: AlertSink | None = None,
    now: float | None = None,
) -> BatchResult:
    """Score one batch of pending samples for a (tenant, route).

    Runs when `min_samples` are waiting, or when the oldest has waited `max_age` seconds. The claim is
    one atomic UPDATE, so concurrent batches never score the same sample. The judge-call cap is checked
    between samples: a sample in progress finishes, the rest go back to pending. Samples claimed more than
    `lease` seconds ago (a crashed worker, or a failed save) are returned to pending first.
    """
    now = time.time() if now is None else now
    store.reclaim_stale(tenant_id, route, now - lease)
    pending, oldest = store.pending_info(tenant_id, route)
    if pending == 0:
        return BatchResult(skipped=True, reason="no pending samples")
    if pending < min_samples and now - (oldest or now) < max_age:
        return BatchResult(skipped=True, reason=f"only {pending} pending (< {min_samples}) and not old enough",
                           left_pending=pending)  # fmt: skip

    token, samples = store.claim_samples(tenant_id, route, limit, now)
    if not samples:
        return BatchResult(skipped=True, reason="another batch claimed the samples")

    counting = CountingJudge(judge)
    cfg = config_hash(judge, embedder)
    faith, relevance, precision = [], [], []
    failed, processed, released, scored = 0, 0, 0, 0
    for sample in samples:
        if counting.calls >= max_judge_calls:
            released = store.release(tenant_id, token)
            break
        processed += 1
        try:
            f = await faithfulness(sample.answer, sample.chunks, counting)
            rel = await answer_relevance(sample.query, sample.answer, counting, embedder)
            prec = await context_precision(sample.query, sample.chunks, counting)
        except Exception as exc:  # JudgeError, or a provider/network error: record it, keep the batch going
            store.mark_failed(tenant_id, sample.sample_id, token, f"{type(exc).__name__}: {exc}")
            failed += 1
            continue
        low = f.score is not None and f.score <= ATTRIBUTION_THRESHOLD
        try:
            store.save_scores(
                tenant_id, sample.sample_id, token,
                faithfulness=f.score, answer_relevance=rel, context_precision=prec,
                verdicts=[v.model_dump() for v in f.verdicts],
                attribution=f.unsupported_claims if low else None, config_hash=cfg,
            )  # fmt: skip
        except Exception:
            # The score is lost, not the sample: it stays claimed and is reclaimed after the lease.
            log.exception("hook2: could not save scores for %s", sample.sample_id)
            failed += 1
            continue
        scored += 1
        if f.score is not None:
            faith.append(f.score)
        if rel is not None:
            relevance.append(rel)
        precision.append(prec)

    if processed == 0:
        return BatchResult(skipped=True, reason="judge-call cap reached before the first sample",
                           left_pending=released)  # fmt: skip
    record = BatchRecord(
        batch_id=uuid.uuid4().hex, tenant_id=tenant_id, route=route, created_at=now, config_hash=cfg,
        scored=scored, failed=failed,
        faithfulness_mean=_mean(faith), faithfulness_min=min(faith) if faith else None,
        answer_relevance_mean=_mean(relevance), context_precision_mean=_mean(precision),
        judge_calls=counting.calls,
    )  # fmt: skip
    # Baseline is read before this batch is stored, so the current batch can never be in it.
    baseline = [b for b in store.recent_batches(tenant_id, route, BASELINE_BATCHES, cfg) if b.scored > 0]
    record.drift = check_drift(baseline, record) if record.scored else []
    store.add_batch(record)
    if sink:
        for alert in record.drift:
            sink.send(alert)
    return BatchResult(skipped=False, batch=record, left_pending=released, alerts=record.drift)
