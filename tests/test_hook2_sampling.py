import asyncio
import time

from rag_quality_monitor import Store
from rag_quality_monitor.hook2 import SampleCapture, Sampler, check_raw_chunks
from rag_quality_monitor.hook2.capture import suspected_intermediate_answer
from rag_quality_monitor.models import RetrievalResult

from .conftest import chunk

RAW = [chunk("c1", "Audit logs are retained for 365 days. Older log entries are removed automatically.")]
ANSWER = "Audit logs are retained for 365 days."


def test_sampler_is_deterministic_per_sample_id():
    s = Sampler(default_rate=0.3)
    first = [s.should_sample(f"id-{i}", "help") for i in range(200)]
    assert first == [Sampler(default_rate=0.3).should_sample(f"id-{i}", "help") for i in range(200)]


def test_sampler_rate_is_roughly_the_configured_rate():
    s = Sampler(default_rate=0.2)
    hits = sum(s.should_sample(f"id-{i}", "help") for i in range(2000))
    assert 300 < hits < 500


def test_sampler_sensitive_route_is_sampled_at_100_percent():
    s = Sampler(default_rate=0.02, route_rates={"compliance": 1.0})
    assert all(s.should_sample(f"id-{i}", "compliance") for i in range(500))
    assert sum(s.should_sample(f"id-{i}", "help") for i in range(500)) < 50


def test_sampler_zero_rate_samples_nothing():
    assert not any(Sampler(default_rate=0.0, route_rates={}).should_sample(f"i{i}", "x") for i in range(100))


async def test_capture_persists_a_raw_sample_scoped_to_its_tenant(store):
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    assert cap.capture("t1", "r", "q", RAW, ANSWER, sample_id="s1") == "queued"
    await cap.drain()
    assert store.get_sample("t1", "s1")["chunks"][0]["id"] == "c1"
    assert store.get_sample("t2", "s1") is None


async def test_capture_does_not_block_the_caller_when_the_store_is_slow():
    class Slow(Store):
        def add_sample(self, sample):
            time.sleep(0.4)
            return super().add_sample(sample)

    store = Slow()
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    start = time.perf_counter()
    cap.capture("t", "r", "q", RAW, ANSWER, sample_id="s1")
    assert time.perf_counter() - start < 0.1
    await asyncio.sleep(0.01)  # the event loop is free while the write runs on a worker thread
    assert store.pending_info("t", "r")[0] == 0
    await cap.drain()
    assert store.pending_info("t", "r")[0] == 1


async def test_capture_never_raises_when_the_store_fails():
    class Failing(Store):
        def add_sample(self, sample):
            raise RuntimeError("db down")

    cap = SampleCapture(Failing(), Sampler(route_rates={"r": 1.0}))
    assert cap.capture("t", "r", "q", RAW, ANSWER, sample_id="s1") == "queued"
    await cap.drain()
    assert cap.counters["failed"] == 1 and cap.counters["persisted"] == 0


def test_capture_without_a_running_loop_does_not_raise(store):
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    assert cap.capture("t", "r", "q", RAW, ANSWER, sample_id="s1") == "error"
    assert cap.counters["failed"] == 1


async def test_queue_overflow_is_dropped_and_counted():
    class Slow(Store):
        def add_sample(self, sample):
            time.sleep(0.05)
            return super().add_sample(sample)

    store = Slow()
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}), max_pending=2)
    results = [cap.capture("t", "r", "q", RAW, ANSWER, sample_id=f"s{i}") for i in range(5)]
    assert results == ["queued", "queued", "dropped", "dropped", "dropped"]
    assert cap.counters["dropped"] == 3
    await cap.drain()
    assert store.pending_info("t", "r")[0] == 2


async def test_sampled_out_samples_are_not_stored(store):
    cap = SampleCapture(store, Sampler(default_rate=0.0, route_rates={}))
    assert cap.capture("t", "r", "q", RAW, ANSWER, sample_id="s1") == "sampled_out"
    await cap.drain()
    assert store.pending_info("t", "r")[0] == 0


FOUND = RetrievalResult(query="q", chunks=RAW, k=3, latency_ms=1.0)


async def test_sample_with_an_intermediate_answer_as_context_is_rejected(store):
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    # The sub-agent's answer was recorded as the "retrieved context": a bare string, no chunk id.
    assert cap.capture("t", "r", "q", [ANSWER], ANSWER, sample_id="s1") == "rejected_not_raw_chunks"
    # Same text dressed up with an id the retriever never returned.
    assert cap.capture("t", "r", "q", [chunk("c9", ANSWER)], ANSWER, "s2", retrieved=FOUND) == "rejected_not_raw_chunks"
    await cap.drain()
    assert cap.counters["rejected_not_raw_chunks"] == 2
    assert store.pending_info("t", "r")[0] == 0


async def test_longer_specialist_answer_with_an_invented_id_is_rejected_by_provenance(store):
    """Longer than the final answer and not a paraphrase of it: the text heuristic cannot see it."""
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    specialist = ("The compliance specialist reviewed the retention policy in detail and concluded that deleted "
                  "records are purged on a rolling schedule after a grace period set by the workspace owner.")  # fmt: skip
    ctx = [chunk("sub-agent-1", specialist)]
    assert cap.capture("t", "r", "q", ctx, ANSWER, "s1") == "queued"  # no provenance given: only shape is checked
    assert cap.counters["provenance_unchecked"] == 1
    assert cap.capture("t", "r", "q", ctx, ANSWER, "s2", retrieved=FOUND) == "rejected_not_raw_chunks"
    assert cap.capture("t", "r", "q", ctx, ANSWER, "s3", retrieved=["c1"]) == "rejected_not_raw_chunks"
    await cap.drain()


async def test_two_sub_agent_answers_as_contexts_are_rejected(store):
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    ctx = [chunk("agent-a", "Agent A says retention is a year."), chunk("agent-b", "Agent B says logs are purged.")]
    assert cap.capture("t", "r", "q", ctx, ANSWER, "s1", retrieved=FOUND) == "rejected_not_raw_chunks"


async def test_known_id_with_altered_text_is_rejected_when_the_full_result_is_given(store):
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    forged = [chunk("c1", "Something a model wrote, filed under a real chunk id.")]
    assert cap.capture("t", "r", "q", forged, ANSWER, "s1", retrieved=FOUND) == "rejected_not_raw_chunks"
    assert cap.capture("t", "r", "q", forged, ANSWER, "s2", retrieved=["c1"]) == "queued"  # ids only: text unchecked
    await cap.drain()


async def test_short_faq_chunk_equal_to_the_extractive_answer_is_accepted_and_at_most_flagged(store):
    faq = chunk("faq1", "Audit logs are retained for 365 days.")
    found = RetrievalResult(query="q", chunks=[faq], k=3, latency_ms=1.0)
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    assert cap.capture("t", "r", "q", [faq], ANSWER, "s1", retrieved=found) == "queued"
    await cap.drain()
    assert store.get_sample("t", "s1") is not None
    assert cap.counters["suspected_intermediate_answer"] == 1  # flagged, not rejected
    assert cap.counters["rejected_not_raw_chunks"] == 0


def test_text_heuristic_flags_containment_both_ways_and_ignores_unrelated_text():
    long_answer = "audit logs are retained for 365 days and older log entries are removed automatically"
    assert suspected_intermediate_answer("Audit logs are retained for 365 days", long_answer)  # context inside answer
    short_answer = "audit logs are retained for 365 days today"
    assert suspected_intermediate_answer(short_answer + " plus some other words", short_answer)  # answer inside short ctx
    assert not suspected_intermediate_answer(short_answer + " " + long_answer + " and more besides", short_answer)  # long ctx
    assert not suspected_intermediate_answer("Guests cannot export documents or create workflows", long_answer)
    assert not suspected_intermediate_answer("too short", long_answer)


def test_text_heuristic_cannot_catch_a_paraphrase():
    assert not suspected_intermediate_answer("Logs from audits are kept for a year before removal", ANSWER)


def test_chunks_without_ids_are_not_raw_chunks():
    assert check_raw_chunks([{"text": "x", "doc_id": "d", "score": 1.0}], "a").chunks is None
    assert check_raw_chunks([{"id": "", "text": "x", "doc_id": "d", "score": 1.0}], "a").chunks is None
    assert check_raw_chunks([{"id": "c", "text": "x", "doc_id": "d", "score": 1.0}], "a").chunks is not None


def test_a_long_chunk_that_contains_a_short_extractive_answer_is_raw_and_not_flagged():
    checked = check_raw_chunks(RAW, ANSWER, FOUND)
    assert checked.chunks is not None and not checked.suspected


async def test_duplicate_sample_ids_are_counted_separately_from_persisted(store):
    cap = SampleCapture(store, Sampler(route_rates={"r": 1.0}))
    for _ in range(3):
        cap.capture("t", "r", "q", RAW, ANSWER, sample_id="same")
    await cap.drain()
    assert (cap.counters["persisted"], cap.counters["duplicate"]) == (1, 2)


async def test_rejection_is_counted_even_when_the_sample_would_have_been_sampled_out(store):
    cap = SampleCapture(store, Sampler(default_rate=0.0, route_rates={}))
    assert cap.capture("t", "r", "q", [ANSWER], ANSWER) == "rejected_not_raw_chunks"
