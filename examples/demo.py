"""End-to-end walkthrough of the Three-Hook Model on a fictional Acme Workspace RAG.

No network, no API keys: the judge and embedder are the deterministic fakes.
Run from the repo root:  python examples/demo.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from acme_rag import AcmeRag  # noqa: E402

from rag_quality_monitor import Chunk, ListAlertSink, Store  # noqa: E402
from rag_quality_monitor.hook1 import Hook1Thresholds, RetrievalMonitor, instrument  # noqa: E402
from rag_quality_monitor.hook2 import SampleCapture, Sampler, run_batch  # noqa: E402
from rag_quality_monitor.hook3 import (  # noqa: E402
    check_thresholds,
    find_regressions,
    load_golden_dir,
    load_thresholds,
    run_suite,
)
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge  # noqa: E402
from rag_quality_monitor.metrics.generation import context_precision  # noqa: E402
from rag_quality_monitor.suite import format_report  # noqa: E402

HERE = Path(__file__).parent
GOLDEN = load_golden_dir(HERE / "golden")
judge, embedder = FakeJudge(), FakeEmbedder()
TENANT = "acme"


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


async def hook1() -> None:
    heading("HOOK 1: heuristic retrieval monitoring (every query, no LLM)")
    store, sink = Store(), ListAlertSink()
    monitor = RetrievalMonitor(store, sink, window=40)
    th = Hook1Thresholds(min_score_gap=0.15, min_events=20)
    questions = [c.question for c in GOLDEN if c.route == "help"]
    for label, rag in (("healthy traffic", AcmeRag()), ("after scores were flattened", AcmeRag(flatten_scores=True))):
        retriever = instrument(rag.retrieve, monitor, "help")
        precisions = []
        for i in range(40):
            result = await retriever(TENANT, questions[i % len(questions)], 5)
            precisions.append(await context_precision(result.query, result.chunks, judge))
        w = monitor.window_stats(TENANT, "help")
        alerts = monitor.check(TENANT, "help", th)
        print(f"\n{label}: 40 queries")
        print(f"  mean score gap {w.mean_score_gap:.3f}   empty rate {w.empty_rate:.0%}   p95 latency {w.latency_p95:.0f} ms")
        print(f"  judged context precision {statistics.fmean(precisions):.2f}")
        print("  alerts:", [a.message for a in alerts] or "none")
    print("\nThe right chunk is still in the top 5, so judged precision looks fine. Only the gap shows it.")


async def hook2() -> None:
    heading("HOOK 2: sampled online evaluation (async, reference-free)")
    store, sink = Store(), ListAlertSink()
    capture = SampleCapture(store, Sampler(default_rate=0.02, route_rates={"compliance": 1.0}))
    questions = [c.question for c in GOLDEN if c.route == "compliance"]
    n = 0
    print("\nbatch  generator   scored  faithfulness  relevance  precision  drift")
    for batch_no in range(1, 6):
        rag = AcmeRag(hallucinate=batch_no == 5)
        for q in questions:
            n += 1
            found = await rag.retrieve(TENANT, q, 5)
            answer = await rag.generate(q, found.chunks)
            capture.capture(TENANT, "compliance", q, found.chunks, answer, sample_id=f"s-{n:03d}", retrieved=found)
        await capture.drain()
        res = await run_batch(store, judge, embedder, TENANT, "compliance", min_samples=10, sink=sink)
        assert res.batch, res.reason
        b = res.batch
        verdict = "ALERT" if b.drift else ("no verdict yet" if batch_no <= 3 else "ok")
        print(f"{batch_no:>5}  {'degraded' if batch_no == 5 else 'healthy':<10}  {b.scored:>6}  "
              f"{b.faithfulness_mean:>12.2f}  {b.answer_relevance_mean:>9.2f}  {b.context_precision_mean:>9.2f}  {verdict}")  # fmt: skip
    print("\ndrift alerts:")
    for a in sink.alerts:
        print("  -", a.message)

    print("\nDrill-down on the last (hallucinated) sample:")
    s = store.get_sample(TENANT, f"s-{n:03d}")
    print(f"  question: {s['query']}")
    print(f"  faithfulness: {s['faithfulness']:.2f}   raw chunks kept: {[c['id'] for c in s['chunks']]}")
    for v in s["verdicts"]:
        print(f"  [{'supported  ' if v['supported'] else 'UNSUPPORTED'}] {v['claim']}")
    print("  claim attribution (faithfulness <= 0.80):", s["unsupported_claims"])

    print("\nPipeline bugs record an intermediate answer as the context:")
    q = "How long are audit logs retained?"
    found = await AcmeRag().retrieve(TENANT, q, 5)
    specialist = "The retention specialist concluded that audit records are purged on a rolling schedule."
    for label, contexts in (
        ("a bare answer string", [specialist]),
        ("an answer under an id the retriever never returned", [Chunk(id="agent-1", doc_id="agent", text=specialist, score=1.0)]),
    ):
        print(f"  {label:<52} -> {capture.capture(TENANT, 'compliance', q, contexts, specialist, retrieved=found)}")
    print(f"  counters: rejected_not_raw_chunks={capture.counters['rejected_not_raw_chunks']}")
    print("  (provenance is the real guard; the text heuristic only flags, and cannot see a paraphrase)")


async def hook3() -> None:
    heading("HOOK 3: offline regression suite (CI gate)")
    thresholds = load_thresholds(HERE / "thresholds.json")
    good = AcmeRag()
    baseline = await run_suite(GOLDEN, good.retrieve, good.generate, judge, embedder)
    print("\nGood pipeline:\n")
    print(format_report(baseline, check_thresholds(baseline, thresholds), []))

    broken = AcmeRag(drop_chunk_ids={"h2", "h4", "h9", "c1", "c2", "c6"})
    report = await run_suite(GOLDEN, broken.retrieve, broken.generate, judge, embedder)
    print("\nAfter a re-index that dropped six relevant chunks:\n")
    print(format_report(report, check_thresholds(report, thresholds), find_regressions(report, baseline, thresholds)))


async def main() -> None:
    await hook1()
    await hook2()
    await hook3()


if __name__ == "__main__":
    asyncio.run(main())
