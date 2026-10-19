"""Hook 1, over time: record every retrieval, watch a rolling window, alert on breaches."""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from ..alerts import AlertSink
from ..models import Alert, RetrievalResult
from ..store import Store
from .stats import RetrievalStats, retrieval_stats

log = logging.getLogger(__name__)

Retriever = Callable[[str, str, int], Awaitable[RetrievalResult]]  # (tenant_id, query, k)


class RetrievalEvent(BaseModel):
    tenant_id: str
    route: str
    stats: RetrievalStats
    ts: float


class WindowStats(BaseModel):
    events: int
    empty_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    mean_score_gap: float | None  # None when no query returned two or more chunks
    mean_fill: float  # mean of count / k: how full the top-k lists are


class Hook1Thresholds(BaseModel):
    # Simplification: absolute gap threshold per route; learn it from a healthy baseline window instead.
    min_score_gap: float = 0.15
    max_empty_rate: float = 0.10
    max_p95_latency_ms: float = 500.0
    min_events: int = 20


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: simple, and always a value that was actually observed."""
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)) - 1, 0)]


class RetrievalMonitor:
    def __init__(
        self,
        store: Store,
        sink: AlertSink | None = None,
        window: int = 200,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store, self.sink, self.window, self.clock = store, sink, window, clock

    def record(self, tenant_id: str, route: str, stats: RetrievalStats) -> None:
        # Simplification: one synchronous INSERT per query; batch the writes if this ever shows in a profile.
        event = RetrievalEvent(tenant_id=tenant_id, route=route, stats=stats, ts=self.clock())
        self.store.add_event(tenant_id, route, event.ts, stats.model_dump_json())

    def window_stats(self, tenant_id: str, route: str) -> WindowStats | None:
        rows = self.store.recent_events(tenant_id, route, self.window)
        if not rows:
            return None
        stats = [RetrievalStats.model_validate_json(s) for _, s in rows]
        latencies = [s.latency_ms for s in stats]
        gaps = [s.score_gap for s in stats if s.count >= 2]  # a gap needs two scores
        return WindowStats(
            events=len(stats),
            empty_rate=sum(s.empty for s in stats) / len(stats),
            latency_p50=percentile(latencies, 50),
            latency_p95=percentile(latencies, 95),
            latency_p99=percentile(latencies, 99),
            mean_score_gap=statistics.fmean(gaps) if gaps else None,
            mean_fill=statistics.fmean([min(s.count / s.k, 1.0) for s in stats if s.k > 0] or [0.0]),
        )

    def check(self, tenant_id: str, route: str, thresholds: Hook1Thresholds) -> list[Alert]:
        w = self.window_stats(tenant_id, route)
        if w is None or w.events < thresholds.min_events:
            return []
        alerts = []

        def add(kind: str, msg: str, value: float, limit: float) -> None:
            alerts.append(Alert(kind=kind, tenant_id=tenant_id, route=route, message=msg, value=value, threshold=limit))

        if w.mean_score_gap is not None and w.mean_score_gap < thresholds.min_score_gap:
            add("score_gap_collapse", f"mean score gap {w.mean_score_gap:.3f} < {thresholds.min_score_gap}",
                w.mean_score_gap, thresholds.min_score_gap)  # fmt: skip
        if w.empty_rate > thresholds.max_empty_rate:
            add("empty_rate", f"empty-result rate {w.empty_rate:.0%} > {thresholds.max_empty_rate:.0%}",
                w.empty_rate, thresholds.max_empty_rate)  # fmt: skip
        if w.latency_p95 > thresholds.max_p95_latency_ms:
            add("latency_p95", f"p95 latency {w.latency_p95:.0f} ms > {thresholds.max_p95_latency_ms:.0f} ms",
                w.latency_p95, thresholds.max_p95_latency_ms)  # fmt: skip
        if self.sink:
            for alert in alerts:
                self.sink.send(alert)
        return alerts


def instrument(retriever: Retriever, monitor: RetrievalMonitor, route: str) -> Retriever:
    """Wrap a retriever so every call is recorded. Monitoring failures are logged, never raised:
    a broken monitor must not take down search."""

    async def wrapped(tenant_id: str, query: str, k: int = 5) -> RetrievalResult:
        result = await retriever(tenant_id, query, k)
        try:
            monitor.record(tenant_id, route, retrieval_stats(result))
        except Exception:
            log.exception("hook1: could not record retrieval event")
        return result

    return wrapped
