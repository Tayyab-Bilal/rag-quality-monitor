# Evaluating RAG in production: four dimensions and the Three-Hook Model

*Design notes by Tayyab Bilal. Written while planning the evaluation pipeline for a
multi-tenant RAG system and reviewing its first implementation. This repo implements the design.*

---

## 1. Why RAG evaluation goes wrong

Most teams "add evaluation" by wiring a library such as RAGAS behind an endpoint, running it a few
times on hand-picked examples, and reporting an average. That gives you a number, but it doesn't
give you a quality system. In practice I kept seeing the same failure modes:

- **Nothing fires automatically.** Evaluation runs only when someone calls it, so real traffic is
  never measured and regressions are found by users.
- **One averaged number.** Per-sample scores and per-claim verdicts are thrown away, so when the
  average drops nobody can see *which* answer, or which claim, caused it.
- **Everything is an LLM judgment.** No metric checks retrieval against a known answer (was the
  right chunk returned at all?), so a judge can rate an answer "relevant" while retrieval has
  quietly collapsed.
- **The judge sees the wrong input.** If the pipeline records an intermediate model's *answer* as
  the "retrieved context", then context and faithfulness scores compare one answer with another
  and say nothing about retrieval. This is easy to do by accident when retrieval sits behind
  sub-agents.
- **No golden set, no gate.** Without a curated set of questions with known-good evidence, there is
  nothing to run in CI and nothing to stop a bad change from merging.

The fix isn't a better metric. It's being clear about *what* is being evaluated and *where* each
check runs.

## 2. Four dimensions of RAG quality

A RAG answer can fail in two places (retrieval, generation), and you need two kinds of evidence
(live traffic, curated cases). That gives four dimensions.

| # | Dimension | Question it answers | Needs a reference? |
|---|---|---|---|
| 1 | **Chunk retrieval** | Did we fetch the right evidence, ranked well? | Optional (stronger with one) |
| 2 | **Answer generation** | Is the answer grounded in that evidence and does it address the question? | No |
| 3 | **Offline regression** | Did this change make quality worse? | Yes |
| 4 | **Golden dataset** | What does "correct" look like for our corpus? | It *is* the reference |

### 2.1 Chunk retrieval

Two families of metrics, and you want both:

- **Algorithmic (cheap, deterministic, needs labelled relevant chunks):** Recall@k, Precision@k,
  MRR (rank of the first relevant chunk), nDCG@k (rank-weighted relevance), hit rate. These answer
  "was the right chunk in the top k?" with no model involved.
- **LLM-judged (no labels needed):** context precision (the judge votes on each retrieved chunk's
  relevance to the question, aggregated rank-aware) and, when a reference answer exists, context
  recall (the reference is split into claims; the judge checks each claim is supported by some
  chunk).

Algorithmic metrics are the ground truth when you have labels. Judged metrics extend coverage to
traffic you never labelled.

### 2.2 Answer generation

- **Faithfulness:** split the answer into atomic claims; the judge marks each claim supported or
  unsupported by the retrieved chunks. Score = supported / total. Keep the verdicts: the
  unsupported claims *are* the diagnosis.
- **Answer relevance:** does the answer address the question? One reliable pattern is to have the
  judge generate the questions the answer would satisfy, embed them, and compare with the real
  question by cosine similarity. A direct judge rating is the simpler alternative.

Both are reference-free, which is what makes them usable on live traffic.

### 2.3 Offline regression and 2.4 the golden dataset

A golden case is: question, the chunk ids that should be retrieved, and (optionally) a reference
answer. Offline regression runs the pipeline over the golden set and compares the scores with
thresholds and with the previous run.

**Store the golden set as JSONL files in the repository, not as database rows behind an API.**
Files are version-controlled, reviewed in pull requests alongside the change that needs them, and
runnable in a CI sandbox with no database. A database is fine for *collecting candidates* from
production, but the gate should read files.

## 3. The Three-Hook Model

The four dimensions say *what* to measure. The three hooks say *where* each check runs, at what
cost and how often.

```
                 ┌──────────────────────── every query ───────────────────────┐
  query ──► retrieve ──► generate ──► answer
               │            │
               ▼            ▼
        Hook 1: heuristics   Hook 2: sampled judge (async)        Hook 3: offline suite (CI)
        no LLM, ~free        1–5% of traffic (100% for            golden JSONL, thresholds,
        every query          sensitive routes), reference-free    run on every PR / nightly
```

### Hook 1: heuristic retrieval monitoring (always on, no LLM)

Computed from data the retriever already returns, on **every** query:

- similarity score distribution (mean, std, min, max)
- **score gap** between the top result and the k-th result
- result count vs requested k, and the **empty-result rate**
- retrieval latency (p50 / p95 / p99)

Why it matters: some degradations are invisible to judged metrics. If every chunk starts scoring
about the same (for example, a prefix added to every chunk at ingestion dominates the embedding),
the right chunk may still be in the top k, so judged precision looks fine, but the retriever has
lost its ability to discriminate. A collapsing score gap shows that within minutes. This hook is
cheap enough to keep on all the time.

### Hook 2: sampled online quality evaluation (asynchronous, reference-free)

- **Sampled:** 1–5% of traffic by default, **100% for sensitive routes** (for example compliance
  questions). The sampling decision is per route, not one global switch.
- **Fire-and-forget:** capture the sample (question, the *raw* retrieved chunks with their ids and
  scores, the final answer) after the answer is sent; never block the user.
- **Batched judging:** a background job scores accumulated samples with the reference-free metrics
  (faithfulness, answer relevance, judged context precision).
- **Drill-down kept:** per-sample scores and per-claim verdicts are stored, and low-faithfulness
  samples get claim attribution (which claims are unsupported).
- **Drift:** compare each batch with a rolling baseline per route and alert on a relative drop.

The single most important rule here: **record the raw retrieved chunks, never an intermediate
answer, as the context.**

### Hook 3: offline regression suite (CI gate)

- Golden JSONL per route (start with 30–50 cases each, grow from production samples Hook 2 flags).
- Runs the retriever (and optionally the generator) over the golden set, computes algorithmic
  retrieval metrics plus judged generation metrics.
- **Threshold assertions** fail the build; a **comparison with the last run** catches slow drift
  that stays above thresholds.
- Deterministic parts (algorithmic metrics) gate merges; judged parts gate with tolerance, because
  a judge is never fully deterministic even at temperature 0.

## 4. Practical concerns

- **Judge bias.** An LLM grading output from its own model family is biased. Use judged scores for
  trends and drill-down, not as ground truth; prefer a different provider for the judge; where it
  matters, use a small ensemble and measure agreement.
- **Cost scales with chunks × claims.** Ten samples with ten chunks and five claims each is
  hundreds of judge calls. Sample, batch, cap per run, and meter judge calls like any other paid AI
  call.
- **Determinism.** Temperature 0 isn't a guarantee. Pin the judge model and prompt version, record
  a config hash with every result, and compare like with like.
- **Per-sample drill-down.** Always store per-sample scores and claim verdicts. An average with no
  drill-down is a number you can't act on.
- **Tenancy and auth.** Evaluation data is user data: scope every sample, score and golden case by
  tenant, and take the tenant from the authenticated identity, never from a request field.

## 5. What each hook costs and catches

| | Hook 1 heuristics | Hook 2 sampled judge | Hook 3 offline suite |
|---|---|---|---|
| Runs | every query | 1–5% (100% sensitive) | every PR / nightly |
| Cost | ~free | judge calls on samples | judge calls on golden set |
| Needs labels | no | no | yes (golden JSONL) |
| Catches | retrieval collapse, empty results, latency | ungrounded answers, irrelevant answers, quality drift on real traffic | regressions before merge |
| Misses | answer quality | issues outside the sample | traffic unlike the golden set |

No single hook is enough; together they cover live retrieval, live answers and changes before they
ship.

## 6. Rollout order

1. **Record the right data first:** raw chunks with ids and scores, per sample. Everything else
   depends on it.
2. **Hook 1:** a few days of work, near-zero running cost, immediate signal.
3. **Golden set:** 30–50 cases per route, then **Hook 3** in CI.
4. **Hook 2:** sampler, async capture, batch judge, drift alerts.
5. Grow the golden set from the low-scoring samples Hook 2 finds.
