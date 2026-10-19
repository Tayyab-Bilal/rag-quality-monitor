import pytest

from rag_quality_monitor import ListAlertSink, Store
from rag_quality_monitor.hook1 import Hook1Thresholds, RetrievalMonitor, instrument, retrieval_stats
from rag_quality_monitor.metrics.generation import context_precision
from rag_quality_monitor.models import RetrievalResult

from .conftest import chunk

TH = Hook1Thresholds(min_score_gap=0.15, max_empty_rate=0.1, max_p95_latency_ms=100, min_events=10)


def result(scores, k=3, latency=10.0):
    return RetrievalResult(query="q", chunks=[chunk(f"c{i}", score=s) for i, s in enumerate(scores)], k=k, latency_ms=latency)


def test_stats_math():
    s = retrieval_stats(result([0.9, 0.5, 0.1, 0.05], k=3))
    assert s.mean == pytest.approx(0.3875)
    assert (s.min, s.max, s.count, s.k, s.empty) == (0.05, 0.9, 4, 3, False)
    assert s.score_gap == pytest.approx(0.8)  # top1 minus the 3rd score, not the last
    assert s.std == pytest.approx((0.471875 / 4) ** 0.5)


def test_stats_gap_uses_last_result_when_fewer_than_k():
    assert retrieval_stats(result([0.9, 0.4], k=5)).score_gap == pytest.approx(0.5)


def test_stats_of_empty_result():
    s = retrieval_stats(result([]))
    assert s.empty and s.count == 0 and s.score_gap == 0.0


async def test_instrument_records_every_call_and_returns_result_unchanged(store):
    monitor = RetrievalMonitor(store)

    async def retriever(tenant, query, k):
        return result([0.9, 0.2])

    wrapped = instrument(retriever, monitor, "help")
    for _ in range(5):
        assert (await wrapped("t1", "q", 3)).chunks[0].id == "c0"
    assert monitor.window_stats("t1", "help").events == 5
    assert monitor.window_stats("t2", "help") is None  # tenant scoped


async def test_instrument_never_raises_when_the_store_is_broken():
    class Broken(Store):
        def add_event(self, *a, **k):
            raise RuntimeError("disk full")

    async def retriever(tenant, query, k):
        return result([0.9, 0.2])

    wrapped = instrument(retriever, RetrievalMonitor(Broken()), "help")
    assert (await wrapped("t", "q", 3)).chunks  # the caller still gets its answer


def feed(monitor, results, route="r"):
    for r in results:
        monitor.record("t", route, retrieval_stats(r))


def test_no_alerts_on_healthy_traffic(store):
    monitor = RetrievalMonitor(store)
    feed(monitor, [result([0.9, 0.5, 0.2])] * 20)
    assert monitor.check("t", "r", TH) == []


def test_too_few_events_gives_no_verdict(store):
    monitor = RetrievalMonitor(store)
    feed(monitor, [result([])] * 3)
    assert monitor.check("t", "r", TH) == []


def test_empty_result_rate_alert(store):
    monitor = RetrievalMonitor(store)
    feed(monitor, [result([0.9, 0.5, 0.2])] * 15 + [result([])] * 5)
    assert [a.kind for a in monitor.check("t", "r", TH)] == ["empty_rate"]


def test_p95_latency_alert_and_percentiles(store):
    sink = ListAlertSink()
    monitor = RetrievalMonitor(store, sink)
    feed(monitor, [result([0.9, 0.5, 0.2], latency=10)] * 17 + [result([0.9, 0.5, 0.2], latency=900)] * 3)
    w = monitor.window_stats("t", "r")
    assert (w.latency_p50, w.latency_p95, w.latency_p99) == (10, 900, 900)
    assert [a.kind for a in monitor.check("t", "r", TH)] == ["latency_p95"]
    assert [a.kind for a in sink.alerts] == ["latency_p95"]


async def test_gap_collapse_fires_while_relevant_chunk_is_still_in_top_k_and_judged_precision_misses_it(rag, judge):
    """The design's motivating example."""
    store = Store()
    monitor = RetrievalMonitor(store, window=30)
    q = "How do I archive a project?"

    healthy = instrument(rag.retrieve, monitor, "healthy")
    for _ in range(30):
        res = await healthy("t", q, 5)
    assert monitor.check("t", "healthy", TH) == []
    healthy_precision = await context_precision(q, res.chunks, judge)

    from acme_rag import AcmeRag

    flat = instrument(AcmeRag(flatten_scores=True).retrieve, monitor, "flat")
    for _ in range(30):
        res = await flat("t", q, 5)
    assert "h2" in [c.id for c in res.chunks]  # the right chunk is still retrieved
    assert [a.kind for a in monitor.check("t", "flat", TH)] == ["score_gap_collapse"]
    assert await context_precision(q, res.chunks, judge) == healthy_precision == 1.0


def test_gap_mean_ignores_single_result_queries(store):
    monitor = RetrievalMonitor(store)
    feed(monitor, [result([0.9, 0.5, 0.2])] * 10 + [result([0.9])] * 10)
    w = monitor.window_stats("t", "r")
    assert w.mean_score_gap == pytest.approx(0.7)  # the single-result queries do not drag it to ~0.35
    assert monitor.check("t", "r", TH) == []


def test_all_single_result_queries_give_no_gap_and_no_gap_alert(store):
    monitor = RetrievalMonitor(store)
    feed(monitor, [result([0.9])] * 20)
    assert monitor.window_stats("t", "r").mean_score_gap is None
    assert monitor.check("t", "r", TH) == []


def test_window_reports_how_full_the_top_k_lists_are(store):
    monitor = RetrievalMonitor(store)
    feed(monitor, [result([0.9, 0.5, 0.2], k=3)] * 5 + [result([0.9], k=3)] * 5)
    assert monitor.window_stats("t", "r").mean_fill == pytest.approx((1.0 + 1 / 3) / 2)
