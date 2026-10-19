# Usage guide

Every Python block here is executed by `tests/test_docs.py`, from the repo root. Install first:
`uv venv .venv && uv pip install -e ".[dev,api]"`.

## Your pipeline

The library needs two async callables.

- A **retriever**: `async (tenant_id, query, k) -> RetrievalResult`. Return the *raw* chunks, each with an id and a score.
- A **generator**: `async (question, chunks) -> str`.

`examples/acme_rag.py` is a complete small example, with three knobs to break it on purpose:
`flatten_scores`, `drop_chunk_ids` and `hallucinate`.

## Dimension 1 and 2: the metrics

```python
import asyncio

from rag_quality_monitor.judge import FakeEmbedder, FakeJudge
from rag_quality_monitor.metrics.generation import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)
from rag_quality_monitor.metrics.retrieval import hit_rate, precision_at_k
from rag_quality_monitor.models import Chunk

chunks = [
    Chunk(id="c1", doc_id="retention", text="Audit logs are retained for 365 days. Older entries are removed.", score=0.9),
    Chunk(id="c2", doc_id="access", text="Guests cannot export documents.", score=0.2),
]
answer = "Audit logs are retained for 365 days. Every plan includes unlimited phone support."
question = "How long are audit logs retained?"


async def main():
    judge, embedder = FakeJudge(), FakeEmbedder()
    f = await faithfulness(answer, chunks, judge)
    print("faithfulness", f.score, "unsupported:", f.unsupported_claims)
    print("relevance", round(await answer_relevance(question, answer, judge, embedder), 2))
    print("context precision", await context_precision(question, chunks, judge))
    print("context recall", await context_recall("Audit logs are retained for 365 days.", chunks, judge))
    assert f.score == 0.5 and f.unsupported_claims == ["Every plan includes unlimited phone support."]


asyncio.run(main())
print(hit_rate(["c2", "c1"], {"c1"}, 1), precision_at_k(["c2", "c1"], {"c1"}, 2))
```

Judged metrics return the evidence, not only a score: `FaithfulnessResult.verdicts` has one
`ClaimVerdict` per claim, with the id of the chunk that supports it.

## Hook 1: always-on retrieval monitoring

Wrap the retriever once. Run `check()` on a schedule (a cron job or a background loop).

```python
import asyncio
import sys

sys.path.insert(0, "examples")
from acme_rag import AcmeRag

from rag_quality_monitor import ListAlertSink, Store
from rag_quality_monitor.hook1 import Hook1Thresholds, RetrievalMonitor, instrument

store, sink = Store(), ListAlertSink()
monitor = RetrievalMonitor(store, sink, window=100)
retriever = instrument(AcmeRag().retrieve, monitor, route="help")


async def traffic():
    for q in ["How do I archive a project?", "How do I pause a workflow?", "Something unrelated entirely"]:
        for _ in range(10):
            await retriever("tenant-1", q, 5)


asyncio.run(traffic())
window = monitor.window_stats("tenant-1", "help")
print(f"events={window.events} empty_rate={window.empty_rate:.0%} gap={window.mean_score_gap:.2f} p95={window.latency_p95}ms")
alerts = monitor.check("tenant-1", "help", Hook1Thresholds(max_empty_rate=0.1, min_events=20))
print([a.message for a in alerts])
assert [a.kind for a in alerts] == ["empty_rate"]  # a third of the queries matched nothing
```

Thresholds (`Hook1Thresholds`): `min_score_gap`, `max_empty_rate`, `max_p95_latency_ms`, `min_events`.
With fewer than `min_events` events in the window there is no verdict.

## Hook 2: sampled online evaluation

Capture after the answer is sent. Score in batches. Look at one sample when a number drops.

```python
import asyncio
import sys

sys.path.insert(0, "examples")
from acme_rag import AcmeRag

from rag_quality_monitor import ListAlertSink, Store
from rag_quality_monitor.hook2 import Sampler, SampleCapture, run_batch
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge

QUESTIONS = ["How long are audit logs retained?", "When do sessions expire after inactivity?"]


async def main():
    store, sink = Store(), ListAlertSink()
    judge, embedder = FakeJudge(), FakeEmbedder()
    # 2% of normal traffic, every compliance question.
    capture = SampleCapture(store, Sampler(default_rate=0.02, route_rates={"compliance": 1.0}))

    n = 0
    for round_no in range(5):
        rag = AcmeRag(hallucinate=round_no == 4)  # the last round degrades
        for q in QUESTIONS:
            n += 1
            found = await rag.retrieve("tenant-1", q, 5)
            answer = await rag.generate(q, found.chunks)
            capture.capture("tenant-1", "compliance", q, found.chunks, answer, sample_id=f"s{n}", retrieved=found)
        await capture.drain()
        result = await run_batch(store, judge, embedder, "tenant-1", "compliance", min_samples=2, sink=sink)
        print(round_no, result.batch.faithfulness_mean, [a.kind for a in result.alerts])

    sample = store.get_sample("tenant-1", f"s{n}")
    print(sample["unsupported_claims"])
    assert sink.alerts and sink.alerts[0].kind == "quality_drift"
    assert sample["unsupported_claims"]  # claim attribution for the low-faithfulness sample
    print(capture.counters)


asyncio.run(main())
```

Things to know:

- `capture()` returns a status string (`queued`, `sampled_out`, `dropped`, `rejected_not_raw_chunks`, `error`)
  and never raises. It needs a running event loop.
- Pass the retriever's result (or its chunk ids) as `retrieved=`. That is the real guard for "raw chunks,
  never an intermediate answer": any context the retriever did not return is rejected and counted in
  `capture.counters["rejected_not_raw_chunks"]`; with a full `RetrievalResult` the text must match too.
  A bare string or a chunk without an id is always rejected.
- Without `retrieved=`, only that shape check and a text heuristic run (`provenance_unchecked` counts
  those captures). The heuristic compares word 3-grams between each context and the answer and bumps
  `suspected_intermediate_answer`; it flags but never rejects (a short FAQ chunk can equal an extractive
  answer) and it cannot catch a paraphrase.
- `run_batch(..., min_samples, max_age, max_judge_calls, limit)`: a batch runs when `min_samples` are
  pending, or the oldest has waited `max_age` seconds. It stops at `max_judge_calls` and leaves the rest pending.
- Drift needs three earlier batches with the same config hash (judge, embedder, prompt, metrics). It
  compares with the last ten. A drop of exactly 15% is not an alert.
- Samples claimed more than `lease` seconds ago (default 600) are returned to pending, so a crashed worker
  or a failed save does not strand them. A failed judge call marks the sample `eval_failed` instead.
- `answer_relevance` is None, not 0, when the judge produced no questions; it is left out of the aggregates.
- Workers may share one SQLite file; give each its own `Store(path)`.

## Hook 3: the golden set and the CI gate

A golden file is `examples/golden/<route>.jsonl`, one case per line:

```text
{"case_id": "help-02", "question": "How do I archive a project?", "relevant_chunk_ids": ["h2"], "reference_answer": "To archive a project, open its settings and choose Archive."}
```

`reference_answer` is optional. The route is the file name. Errors name the file and line.

```python
import asyncio
import sys

sys.path.insert(0, "examples")
from acme_rag import AcmeRag

from rag_quality_monitor.hook3 import (
    GoldenError,
    check_thresholds,
    find_regressions,
    load_golden_dir,
    load_golden_file,
    load_thresholds,
    run_suite,
)
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge

golden = load_golden_dir("examples/golden")
thresholds = load_thresholds("examples/thresholds.json")


def suite(rag):
    return asyncio.run(run_suite(golden, rag.retrieve, rag.generate, FakeJudge(), FakeEmbedder()))


before = suite(AcmeRag())
after = suite(AcmeRag(drop_chunk_ids={"h2", "h4"}))  # a re-index lost two chunks
print([(v.route, v.metric, v.value) for v in check_thresholds(after, thresholds)])
print([(r.route, r.metric, r.baseline, round(r.current, 2)) for r in find_regressions(after, before, thresholds)][:3])
assert check_thresholds(before, thresholds) == []
assert check_thresholds(after, thresholds)

try:
    load_golden_file("examples/thresholds.json")  # not a golden file
except GoldenError as exc:
    print(exc)
```

The CLI wraps the same calls:

```bash
python -m rag_quality_monitor.suite --golden examples/golden --thresholds examples/thresholds.json \
    --baseline reports/last.json --write reports/last.json
```

Exit codes: 0 pass, 1 threshold failure or regression, 2 bad input (golden file, thresholds, baseline, or
a pipeline that cannot be loaded). `--pipeline` points at a Python
file defining `build_pipeline() -> (retriever, generator)` (default `examples/acme_rag.py`). A missing
baseline file is not an error, so a first run passes and can write one. This repo's CI runs the gate against
the committed `examples/baseline.json`, so the regression half of Hook 3 runs on every push. After an
intended change, regenerate it:

```bash
python -m rag_quality_monitor.suite --golden examples/golden --thresholds examples/thresholds.json \
    --write examples/baseline.json
```

Judged metrics are only compared with a baseline that has the same config hash (judge, embedder, prompt,
`k` and a fingerprint of the golden files); otherwise the CLI prints a note and compares the algorithmic
metrics only.

Thresholds file:

```json
{
  "algorithmic": {"recall_at_k": 0.9, "mrr": 0.8},
  "judged": {"faithfulness": 0.9},
  "judged_tolerance": 0.05
}
```

Algorithmic thresholds are strict. A judged metric passes down to `threshold - judged_tolerance`. A
threshold on a metric that was not measured (for example a judged one with no judge) fails. The file is
validated: unknown keys, unknown metric names (`recall_at_k`, `precision_at_k`, `hit_rate`, `mrr`,
`ndcg_at_k`; `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`) and a file that
defines no metrics are all errors, because a typo must not turn the gate into a no-op.

## Real judges

Install an extra and set the environment; nothing else changes.

```bash
uv pip install -e ".[anthropic]"   # or ".[openai]"
export RQM_JUDGE=anthropic ANTHROPIC_API_KEY=...   # optional: RQM_JUDGE_MODEL=...
```

```python
from rag_quality_monitor.judge import FakeJudge, judge_from_env

judge = judge_from_env()  # the fake unless RQM_JUDGE and the matching key are set
print(type(judge).__name__, judge.model)
assert isinstance(judge, FakeJudge) or judge.model
```

You can also build one directly: `AnthropicJudge(model="claude-opus-5-5")`, `OpenAIJudge(model=...)`,
`OpenAIEmbedder()`. Each takes `client=` so tests can pass a fake. A reply that is not valid JSON, or a
refusal, raises `JudgeError`; `run_batch` records that sample as `eval_failed`.

## HTTP API

```python
from fastapi.testclient import TestClient

from rag_quality_monitor import Store
from rag_quality_monitor.api import FakeTokenVerifier, create_app
from rag_quality_monitor.judge import FakeEmbedder, FakeJudge
from rag_quality_monitor.models import Chunk, Sample

store = Store()
store.add_sample(
    Sample(
        sample_id="s1", tenant_id="tenant-a", route="compliance", query="How long are audit logs retained?",
        chunks=[Chunk(id="c1", doc_id="d", text="Audit logs are retained for 365 days.", score=0.9)],
        answer="Audit logs are retained for 365 days. Pets are allowed.", created_at=1.0,
    )
)
verifier = FakeTokenVerifier({"token-a": "tenant-a", "token-b": "tenant-b"})
client = TestClient(create_app(store, FakeJudge(), FakeEmbedder(), verifier))
a, b = {"Authorization": "Bearer token-a"}, {"Authorization": "Bearer token-b"}

assert client.post("/batches/run", json={"route": "compliance", "min_samples": 1}, headers=a).json()["batch"]["scored"] == 1
sample = client.get("/samples/s1", headers=a).json()
print(sample["faithfulness"], sample["unsupported_claims"])
assert client.get("/samples/s1", headers=b).status_code == 404  # tenant B sees nothing of tenant A
```

Endpoints: `GET /stats?route=`, `GET /samples/{id}`, `POST /batches/run`, `GET /golden/report`. The tenant
always comes from the bearer token. `create_app(..., max_judge_calls=500)` sets the judge-call cap on the
server; a request's `max_judge_calls` can only lower it. `min_samples` defaults to 10, as in the library. Replace `FakeTokenVerifier` with your own `TokenVerifier`.
