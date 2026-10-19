# API reference

Public classes and functions. Import paths are relative to `rag_quality_monitor`.

## Models (`models`)

| Class | Fields |
|---|---|
| `Chunk` | `id: str` (required, non-empty), `doc_id: str`, `text: str`, `score: float` |
| `RetrievalResult` | `query: str`, `chunks: list[Chunk]`, `k: int`, `latency_ms: float` |
| `Sample` | `sample_id`, `tenant_id`, `route`, `query`, `chunks: list[Chunk]`, `answer`, `created_at: float` |
| `GoldenCase` | `case_id`, `route`, `question`, `relevant_chunk_ids: list[str]` (at least one), `reference_answer: str \| None` |
| `ClaimVerdict` | `claim: str`, `supported: bool`, `evidence_chunk_id: str \| None` |
| `Alert` | `kind`, `tenant_id`, `route`, `message`, `value`, `threshold` |
| `BatchRecord` | batch aggregates, `config_hash`, `judge_calls`, `drift: list[Alert]` |

## Alerts (`alerts`)

- `AlertSink` (Protocol): `send(alert: Alert) -> None`.
- `ListAlertSink`: keeps alerts in `.alerts`.

## Store (`store.Store`)

`Store(path: str = ":memory:")`. Every method takes `tenant_id`.

- `add_event`, `recent_events(tenant_id, route, limit)`: Hook 1 events.
- `add_sample(sample)`: insert; a repeated `(tenant, sample_id)` is ignored.
- `pending_info(tenant_id, route) -> (count, oldest_created_at)`
- `claim_samples(tenant_id, route, limit) -> (token, list[Sample])`: atomic claim.
- `release(tenant_id, token) -> int`: return unprocessed claimed samples to pending.
- `save_scores(...)`, `mark_failed(tenant_id, sample_id, token, error)`
- `get_sample(tenant_id, sample_id) -> dict | None`: the drill-down (chunks, answer, scores, verdicts, `unsupported_claims`).
- `count_by_status(tenant_id, route) -> dict[str, int]`
- `add_batch`, `recent_batches(tenant_id, route, limit, config_hash=None)`, `latest_batch`
- `save_suite_report(tenant_id, report_json)`, `latest_suite_report(tenant_id)`

## Retrieval metrics (`metrics.retrieval`)

All take `(retrieved: Sequence[str], relevant: Collection[str], k)` and return `float`.

- `recall_at_k`, `precision_at_k`, `hit_rate`, `ndcg_at_k`
- `mrr(retrieved, relevant, k=None)`

Raise `ValueError` for an empty `relevant` set or non-positive `k`. See [metrics.md](metrics.md).

## Generation metrics (`metrics.generation`)

All async.

- `faithfulness(answer, chunks, judge) -> FaithfulnessResult(score, verdicts, unsupported_claims, reason)`
- `answer_relevance(question, answer, judge, embedder, n=3) -> float | None` (None if no questions were generated)
- `context_precision(question, chunks, judge) -> float`
- `context_recall(reference_answer, chunks, judge) -> float | None`

## Judges (`judge`)

- `Judge` (Protocol, async): `model: str`; `extract_claims(text)`, `verify_claims(claims, chunks)`,
  `chunk_relevant(question, chunk)`, `generate_questions(answer, n)`.
- `Embedder` (Protocol): `async embed(texts) -> list[list[float]]`.
- `FakeJudge(support_threshold=0.6, relevance_threshold=0.6, model="fake-judge-v1")`, `FakeEmbedder(dim=256)`.
- `CountingJudge(judge)`: wraps a judge; `.calls` counts every judge call.
- `JudgeError`: unusable judge reply (bad JSON, refusal).
- `AnthropicJudge(model="claude-opus-5-5", client=None, max_tokens=16000, effort="low")` in `judge.anthropic_judge` (extra `anthropic`).
- `OpenAIJudge(model="gpt-4o-mini", client=None, temperature=<0, or None for o-series/gpt-5>)`, `OpenAIEmbedder(model, client=None)` in `judge.openai_judge` (extra `openai`).
- `judge_from_env()`, `embedder_from_env()`; `PROMPT_VERSION`.

## Hook 1 (`hook1`)

- `retrieval_stats(result) -> RetrievalStats(mean, std, min, max, score_gap, count, k, empty, latency_ms)`
- `instrument(retriever, monitor, route)`: returns a retriever with the same signature that records each call.
- `RetrievalMonitor(store, sink=None, window=200, clock=time.time)`
  - `record(tenant_id, route, stats)`
  - `window_stats(tenant_id, route) -> WindowStats | None`
  - `check(tenant_id, route, thresholds: Hook1Thresholds) -> list[Alert]`
- `WindowStats`: `events`, `empty_rate`, `latency_p50/p95/p99`, `mean_score_gap` (queries with two or more chunks only; None if none), `mean_fill` (mean of count / k).
- `Hook1Thresholds(min_score_gap=0.15, max_empty_rate=0.10, max_p95_latency_ms=500, min_events=20)`

Alert kinds: `score_gap_collapse`, `empty_rate`, `latency_p95`.

## Hook 2 (`hook2`)

- `Sampler(default_rate=0.02, route_rates={"compliance": 1.0})`: `should_sample(sample_id, route)`, `rate(route)`.
- `SampleCapture(store, sampler, max_pending=1000, clock=time.time)`
  - `capture(tenant_id, route, query, contexts, answer, sample_id=None, retrieved=None) -> str`: `retrieved` is the
    `RetrievalResult` (or chunk ids) the contexts must come from; without it provenance is unchecked.
  - `await drain()`; `counters` dict.
- `check_raw_chunks(contexts, answer, retrieved=None) -> RawCheck(chunks | None, reason, suspected)`
- `suspected_intermediate_answer(context, answer) -> bool` (in `hook2.capture`): word 3-gram heuristic; a flag, not a guard.
- `run_batch(store, judge, embedder, tenant_id, route, *, min_samples=10, max_age=3600, lease=600, max_judge_calls=500, limit=50, sink=None, now=None) -> BatchResult`
- `BatchResult(skipped, reason, batch, left_pending, alerts)`
- `config_hash(judge, embedder=None) -> str`
- `check_drift(baseline: list[BatchRecord], current: BatchRecord) -> list[Alert]`: alert kind `quality_drift`.

## Hook 3 (`hook3`)

- `load_golden_file(path)`, `load_golden_dir(directory)`; `GoldenError` messages start with `file:line`.
- `run_suite(golden, retriever, generator=None, judge=None, embedder=None, *, k=5, tenant_id="suite") -> SuiteReport`
- `SuiteReport(k, config_hash, cases: list[CaseResult], routes: dict[str, RouteSummary])`
- `Thresholds(algorithmic, judged, judged_tolerance=0.05)` (unknown keys and metric names are errors), `load_thresholds(path)` (also rejects a file with no metrics)
- `check_thresholds(report, thresholds) -> list[Violation]`
- `find_regressions(report, baseline, thresholds) -> list[Regression]` (judged metrics only when `judged_comparable(report, baseline)`, i.e. the config hashes match)

CLI: `python -m rag_quality_monitor.suite --golden DIR --thresholds FILE [--baseline FILE] [--write FILE] [--pipeline FILE] [--k N]`.

## HTTP (`api`, extra `api`)

- `create_app(store, judge, embedder, verifier, max_judge_calls=500) -> FastAPI`
- `TokenVerifier` (Protocol): `verify(token) -> tenant_id | None`; `FakeTokenVerifier({token: tenant})`.
- Routes: `GET /stats?route=`, `GET /samples/{sample_id}`, `POST /batches/run` (`{"route", "min_samples"=10, "max_judge_calls"}`; the last can only lower the server cap), `GET /golden/report`.
