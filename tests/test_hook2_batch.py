import asyncio
import threading
import uuid

import pytest

from rag_quality_monitor import ListAlertSink, Store
from rag_quality_monitor.hook2 import check_drift, config_hash, run_batch
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge, JudgeError
from rag_quality_monitor.models import BatchRecord, Sample

from .conftest import chunk

TEXT = "Audit logs are retained for 365 days. Older log entries are removed automatically."


def add(store, n, tenant="t", route="r", answer="Audit logs are retained for 365 days.", t0=1000.0):
    for i in range(n):
        store.add_sample(Sample(sample_id=f"s{i}", tenant_id=tenant, route=route, query=f"How long are audit logs retained? #{i}",
                                chunks=[chunk("c1", TEXT)], answer=answer, created_at=t0 + i))  # fmt: skip


async def test_batch_scores_samples_and_stores_per_sample_scores_and_verdicts(store, judge, embedder):
    add(store, 3)
    res = await run_batch(store, judge, embedder, "t", "r", min_samples=3)
    assert res.batch.scored == 3 and res.batch.faithfulness_mean == 1.0
    s = store.get_sample("t", "s0")
    assert s["status"] == "scored" and s["faithfulness"] == 1.0
    assert s["verdicts"][0]["evidence_chunk_id"] == "c1"
    assert s["config_hash"] == config_hash(judge, embedder)


async def test_low_faithfulness_keeps_claim_attribution_and_high_does_not(store, judge, embedder):
    add(store, 1, answer="Audit logs are retained for 365 days. Every plan includes unlimited phone support.")
    store.add_sample(Sample(sample_id="ok", tenant_id="t", route="r", query="q", chunks=[chunk("c1", TEXT)],
                            answer="Audit logs are retained for 365 days.", created_at=2000.0))  # fmt: skip
    await run_batch(store, judge, embedder, "t", "r", min_samples=1)
    bad, good = store.get_sample("t", "s0"), store.get_sample("t", "ok")
    assert bad["faithfulness"] == 0.5
    assert bad["unsupported_claims"] == ["Every plan includes unlimited phone support."]
    assert good["unsupported_claims"] == []


async def test_batch_waits_for_min_samples_unless_the_oldest_is_too_old(store, judge, embedder):
    add(store, 2, t0=1000.0)
    res = await run_batch(store, judge, embedder, "t", "r", min_samples=5, max_age=60, now=1010.0)
    assert res.skipped and store.pending_info("t", "r")[0] == 2
    res = await run_batch(store, judge, embedder, "t", "r", min_samples=5, max_age=60, now=1100.0)
    assert not res.skipped and res.batch.scored == 2


async def test_batch_is_scoped_to_its_tenant(store, judge, embedder):
    add(store, 2, tenant="a")
    add(store, 2, tenant="b")
    await run_batch(store, judge, embedder, "a", "r", min_samples=1)
    assert store.count_by_status("a", "r") == {"scored": 2}
    assert store.count_by_status("b", "r") == {"pending": 2}


class SlowJudge(FakeJudge):
    """Yields to the event loop on every call and records which question it was asked about."""

    def __init__(self):
        super().__init__()
        self.seen: list[str] = []
        self._lock = threading.Lock()

    async def chunk_relevant(self, question, chunk):
        with self._lock:
            self.seen.append(question)
        await asyncio.sleep(0.001)
        return await super().chunk_relevant(question, chunk)


async def test_two_concurrent_batches_in_one_loop_never_score_the_same_sample(store, embedder):
    add(store, 12)
    judge = SlowJudge()
    results = await asyncio.gather(
        *(run_batch(store, judge, embedder, "t", "r", min_samples=1, limit=5) for _ in range(4))
    )
    assert sum(r.batch.scored for r in results if r.batch) == 12
    assert len(judge.seen) == len(set(judge.seen)) == 12


def test_batch_claim_is_atomic_across_racing_workers(tmp_path):
    """Four threads, four separate connections to one SQLite file, released together by a barrier."""
    path = str(tmp_path / "rqm.db")
    seed = Store(path)
    add(seed, 40)
    judge = SlowJudge()
    barrier = threading.Barrier(4)
    scored: list[int] = []

    def worker():
        mine = Store(path)
        barrier.wait()

        async def loop():
            while True:
                res = await run_batch(mine, judge, FakeEmbedder(), "t", "r", min_samples=1, limit=3)
                if res.skipped:
                    return
                scored.append(res.batch.scored)

        asyncio.run(loop())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(scored) == 40
    assert len(judge.seen) == len(set(judge.seen)) == 40  # no sample was judged twice
    assert seed.count_by_status("t", "r") == {"scored": 40}


async def test_claim_never_hands_out_a_sample_twice(store):
    add(store, 6)
    _, first = store.claim_samples("t", "r", 4)
    _, second = store.claim_samples("t", "r", 4)
    assert {s.sample_id for s in first}.isdisjoint({s.sample_id for s in second})
    assert len(first) == 4 and len(second) == 2


async def test_cost_cap_leaves_the_remainder_pending(store, judge, embedder):
    add(store, 10)
    res = await run_batch(store, judge, embedder, "t", "r", min_samples=1, max_judge_calls=8)
    # the cap is checked between samples, so the sample in progress finishes
    assert res.batch.scored < 10 and res.batch.judge_calls >= 8
    counts = store.count_by_status("t", "r")
    assert counts["scored"] == res.batch.scored
    assert counts["pending"] == 10 - res.batch.scored == res.left_pending
    assert "claimed" not in counts


async def test_cost_cap_of_zero_scores_nothing_and_releases_everything(store, judge, embedder):
    add(store, 3)
    res = await run_batch(store, judge, embedder, "t", "r", min_samples=1, max_judge_calls=0)
    assert res.skipped and store.count_by_status("t", "r") == {"pending": 3}


class FailingJudge(FakeJudge):
    async def verify_claims(self, claims, chunks):
        if any("poison" in c for c in claims):
            raise JudgeError("bad json")
        return await super().verify_claims(claims, chunks)


async def test_judge_error_marks_the_sample_eval_failed_and_the_batch_continues(store, embedder):
    add(store, 2)
    store.add_sample(Sample(sample_id="bad", tenant_id="t", route="r", query="q", chunks=[chunk("c1", TEXT)],
                            answer="poison claim here.", created_at=5000.0))  # fmt: skip
    res = await run_batch(store, FailingJudge(), embedder, "t", "r", min_samples=1)
    assert (res.batch.scored, res.batch.failed) == (2, 1)
    bad = store.get_sample("t", "bad")
    assert bad["status"] == "eval_failed" and "bad json" in bad["error"]


def rec(faith, cfg="h", rel=0.5, prec=0.5):
    return BatchRecord(batch_id=uuid.uuid4().hex, tenant_id="t", route="r", created_at=0, config_hash=cfg, scored=10,
                       failed=0, faithfulness_mean=faith, answer_relevance_mean=rel, context_precision_mean=prec)  # fmt: skip


def test_drift_alert_on_a_drop_larger_than_15_percent():
    alerts = check_drift([rec(0.9)] * 5, rec(0.7))
    assert [a.kind for a in alerts] == ["quality_drift"] and "faithfulness" in alerts[0].message


def test_no_drift_alert_on_a_drop_smaller_than_15_percent():
    assert check_drift([rec(1.0)] * 5, rec(0.86)) == []


def test_no_verdict_with_fewer_than_3_baseline_batches():
    assert check_drift([rec(0.9)] * 2, rec(0.1)) == []


def test_no_drift_alert_across_different_config_hashes():
    assert check_drift([rec(0.9, cfg="old")] * 5, rec(0.1, cfg="new")) == []
    assert len(check_drift([rec(0.9, cfg="old")] * 2 + [rec(0.9, cfg="new")] * 3, rec(0.1, cfg="new"))) == 1


def test_drift_baseline_is_the_last_10_batches_only():
    old_good, recent_bad = [rec(1.0)] * 5, [rec(0.5)] * 10
    assert check_drift(recent_bad + old_good, rec(0.5)) == []  # baseline is the ten 0.5 batches


async def test_drift_end_to_end_through_run_batch(store, embedder):
    judge, sink = FakeJudge(), ListAlertSink()
    for n in range(4):  # four healthy batches build the baseline
        for i in range(4):
            store.add_sample(Sample(sample_id=f"b{n}-{i}", tenant_id="t", route="r", query="How long are audit logs retained?",
                                    chunks=[chunk("c1", TEXT)], answer="Audit logs are retained for 365 days.", created_at=n))  # fmt: skip
        assert (await run_batch(store, judge, embedder, "t", "r", min_samples=1, sink=sink)).alerts == []
    add(store, 4, answer="Audit logs are retained for 365 days. Every plan includes unlimited phone support.")
    res = await run_batch(store, judge, embedder, "t", "r", min_samples=1, sink=sink)
    assert any("faithfulness" in a.message for a in res.alerts)
    assert sink.alerts == res.alerts
    assert store.latest_batch("t", "r").drift == res.alerts


@pytest.mark.parametrize("other", ["fake-judge-v2"])
def test_config_hash_changes_with_the_judge_model(other):
    assert config_hash(FakeJudge()) != config_hash(FakeJudge(model=other))


def test_config_hash_includes_the_embedder_identity():
    assert config_hash(FakeJudge(), FakeEmbedder(dim=256)) != config_hash(FakeJudge(), FakeEmbedder(dim=128))
    assert config_hash(FakeJudge(), FakeEmbedder()) != config_hash(FakeJudge())


async def test_a_different_embedder_gives_a_different_hash_and_no_cross_hash_drift(store):
    judge, sink = FakeJudge(), ListAlertSink()
    for n in range(4):
        add_round(store, n, "Audit logs are retained for 365 days.")
        await run_batch(store, judge, FakeEmbedder(dim=256), "t", "r", min_samples=1, sink=sink)
    add_round(store, 9, "Audit logs are retained for 365 days. Every plan includes unlimited phone support.")
    res = await run_batch(store, judge, FakeEmbedder(dim=128), "t", "r", min_samples=1, sink=sink)
    assert res.alerts == []  # the baseline used another embedder, so there is no verdict


def add_round(store, n, answer, count=3):
    for i in range(count):
        store.add_sample(Sample(sample_id=f"r{n}-{i}", tenant_id="t", route="r", query="How long are audit logs retained?",
                                chunks=[chunk("c1", TEXT)], answer=answer, created_at=float(n)))  # fmt: skip


async def test_attribution_is_kept_at_exactly_080_and_dropped_just_above(store, judge, embedder):
    four_of_five = "Audit logs are retained for 365 days. " * 4 + "Pets are allowed everywhere."
    store.add_sample(Sample(sample_id="edge", tenant_id="t", route="r", query="q", chunks=[chunk("c1", TEXT)],
                            answer=four_of_five, created_at=1.0))  # fmt: skip
    await run_batch(store, judge, embedder, "t", "r", min_samples=1)
    s = store.get_sample("t", "edge")
    assert s["faithfulness"] == 0.8 and s["unsupported_claims"] == ["Pets are allowed everywhere."]


def test_drift_of_exactly_15_percent_is_not_an_alert():
    assert check_drift([rec(1.0)] * 5, rec(0.85)) == []
    assert len(check_drift([rec(1.0)] * 5, rec(0.849))) == 1


class FlakySaveStore(Store):
    """The judge works but the database refuses the write, as during a failover."""

    fail = True

    def save_scores(self, *a, **k):
        if self.fail:
            raise RuntimeError("database is read-only")
        return super().save_scores(*a, **k)


async def test_failed_save_strands_nothing_samples_are_reclaimed_after_the_lease(embedder, judge):
    store = FlakySaveStore()
    add(store, 3)
    first = await run_batch(store, judge, embedder, "t", "r", min_samples=1, now=1000.0)
    assert first.batch.scored == 0 and first.batch.failed == 3  # one failure did not stop the batch
    assert store.count_by_status("t", "r") == {"claimed": 3}
    store.fail = False
    early = await run_batch(store, judge, embedder, "t", "r", min_samples=1, now=1100.0)  # lease not over
    assert early.skipped
    later = await run_batch(store, judge, embedder, "t", "r", min_samples=1, now=1000.0 + 601)
    assert later.batch.scored == 3 and store.count_by_status("t", "r") == {"scored": 3}


class NoQuestionsJudge(FakeJudge):
    async def generate_questions(self, answer, n):
        return []


async def test_answer_relevance_is_excluded_from_aggregates_when_no_questions_are_generated(store, embedder):
    add(store, 2)
    res = await run_batch(store, NoQuestionsJudge(), embedder, "t", "r", min_samples=1)
    assert res.batch.answer_relevance_mean is None  # not 0.0
    assert res.batch.scored == 2 and res.batch.faithfulness_mean == 1.0
    assert store.get_sample("t", "s0")["answer_relevance"] is None
