# Metrics

Formulas and worked examples for every metric. The code blocks run as part of the test suite.

## Retrieval metrics (algorithmic)

Inputs: the ranked list of retrieved chunk ids, the set of relevant ids, and `k`.

Conventions: duplicate ids count once at their first position. Only the first `k` distinct ids are
used. Precision always divides by `k`. An empty relevant set raises `ValueError` (the case is unlabelled).

| Metric | Formula | Meaning |
|---|---|---|
| recall@k | `|relevant ∩ top_k| / |relevant|` | How much of the needed evidence came back |
| precision@k | `|relevant ∩ top_k| / k` | How much of the top k is useful |
| hit rate | `1` if `relevant ∩ top_k` is not empty, else `0` | Was anything useful there at all (mean over cases) |
| MRR | `1 / rank` of the first relevant id, else `0` | How early the first useful chunk is |
| nDCG@k | `DCG / IDCG`, `DCG = Σ rel_i / log2(i + 1)` | Rank-weighted; relevance is binary |

`IDCG` is the DCG of a perfect ranking: `Σ 1 / log2(i + 1)` for `i = 1..min(|relevant|, k)`.

Worked example. Retrieved `[x, a, y, b, z]`, relevant `{a, b, q}`, `k = 5`:

- recall@5 = 2 / 3 = 0.667 (`q` was never found)
- precision@5 = 2 / 5 = 0.4
- hit rate = 1
- MRR = 1 / 2 = 0.5 (`a` is at rank 2)
- DCG = 1/log2(3) + 1/log2(5) = 0.631 + 0.431 = 1.062; IDCG = 1 + 0.631 + 0.5 = 2.131; nDCG@5 = 0.498

```python
import math

from rag_quality_monitor.metrics.retrieval import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k

retrieved, relevant = ["x", "a", "y", "b", "z"], {"a", "b", "q"}
assert round(recall_at_k(retrieved, relevant, 5), 3) == 0.667
assert precision_at_k(retrieved, relevant, 5) == 0.4
assert hit_rate(retrieved, relevant, 5) == 1.0
assert mrr(retrieved, relevant) == 0.5
dcg = 1 / math.log2(3) + 1 / math.log2(5)
idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
assert math.isclose(ndcg_at_k(retrieved, relevant, 5), dcg / idcg)
print(round(dcg / idcg, 3))
```

## Faithfulness (judged, reference-free)

1. The judge splits the answer into atomic claims.
2. The judge marks each claim supported or unsupported by the retrieved chunks, naming the chunk.
3. `score = supported / total`. No claims means no score (`None`), not a perfect one.

The verdicts are kept. The unsupported claims are the diagnosis, and `run_batch` stores them as claim
attribution when the score is 0.80 or lower.

Worked example: three claims, the third unsupported, gives 2 / 3 = 0.667.

```python
import asyncio

from rag_quality_monitor.judge import FakeJudge
from rag_quality_monitor.metrics.generation import faithfulness
from rag_quality_monitor.models import Chunk

chunks = [Chunk(id="c1", doc_id="d", text="Sessions expire after 12 hours of inactivity.", score=1.0)]
answer = "Sessions expire after 12 hours of inactivity. Sessions expire after 12 hours of inactivity. Pets are allowed."
result = asyncio.run(faithfulness(answer, chunks, FakeJudge()))
print(round(result.score, 3), result.unsupported_claims)
assert round(result.score, 3) == 0.667 and result.unsupported_claims == ["Pets are allowed."]
```

## Answer relevance (judged, reference-free)

The judge writes `n` (default 3) questions the answer would satisfy. All of them and the real question
are embedded. `score = mean cosine(real question, generated question)`. Close to 1 means the answer
addresses the question; near 0 means it answers something else.

If the judge produces no questions the result is `None`, not 0, and the sample is left out of the aggregate.

Worked example: generated questions with cosines 0.8, 0.6, 0.4 give 0.6.

```python
import asyncio

from rag_quality_monitor.judge import FakeEmbedder, FakeJudge
from rag_quality_monitor.metrics.generation import answer_relevance

q = "How long are audit logs retained?"
on = asyncio.run(answer_relevance(q, "Audit logs are retained for 365 days.", FakeJudge(), FakeEmbedder()))
off = asyncio.run(answer_relevance(q, "Guests cannot export documents.", FakeJudge(), FakeEmbedder()))
print(round(on, 2), round(off, 2))
assert on > 0.4 > off
```

## Context precision (judged, no labels)

The judge votes on each retrieved chunk's relevance to the question, in rank order. The score is the
average of `precision@i` taken at each relevant position `i`:

`AP = (Σ over relevant i of  hits_so_far(i) / i) / total_hits`, or 0 if nothing is relevant.

Rank matters: one relevant chunk at rank 1 scores 1.0; at rank 3 it scores 1/3.

```python
import asyncio

from rag_quality_monitor.judge import FakeJudge
from rag_quality_monitor.metrics.generation import context_precision
from rag_quality_monitor.models import Chunk

good = Chunk(id="c1", doc_id="d", text="Audit logs are retained for 365 days.", score=1.0)
bad = Chunk(id="c2", doc_id="d", text="Guests cannot export documents.", score=0.1)
q = "How long are audit logs retained?"
first = asyncio.run(context_precision(q, [good, bad, bad], FakeJudge()))
last = asyncio.run(context_precision(q, [bad, bad, good], FakeJudge()))
print(first, round(last, 3))
assert first == 1.0 and round(last, 3) == 0.333
```

Two relevant chunks at ranks 1 and 3: `(1/1 + 2/3) / 2 = 0.833`.

## Context recall (judged, needs a reference answer)

The reference answer is split into claims; the judge checks each against the chunks.
`score = supported claims / claims`. It answers: did retrieval bring back enough to write the reference?

Worked example: the reference has three claims and the chunks support two of them, so recall is 2 / 3.

```python
import asyncio

from rag_quality_monitor.judge import FakeJudge
from rag_quality_monitor.metrics.generation import context_recall
from rag_quality_monitor.models import Chunk

chunks = [
    Chunk(id="c1", doc_id="d", text="Audit logs are retained for 365 days.", score=0.9),
    Chunk(id="c2", doc_id="d", text="Sessions expire after 12 hours of inactivity.", score=0.4),
]
reference = "Audit logs are retained for 365 days. Sessions expire after 12 hours of inactivity. Pets are allowed."
recall = asyncio.run(context_recall(reference, chunks, FakeJudge()))
print(round(recall, 3))
assert round(recall, 3) == 0.667
```

## Reading the demo numbers

The demo and CI use the fake judge and embedder, so some values look low on purpose.

- **Answer relevance is about 0.43 even for a perfect answer.** The fake embedder is a hashed bag of
  words and the fake judge's questions are word lists from the answer. A short question shares few words
  with a longer answer, so cosine similarity stays well below 1. Read the change, not the level. A real
  embedder reads higher.
- **precision@k is at most 1/k with one labelled relevant chunk** (0.2 at `k = 5`), so it is reported but
  not gated in `examples/thresholds.json`.
- **Context precision ignores irrelevant chunks ranked after the relevant one.** Average precision only
  samples the precision at relevant positions, so noise at the end of the list does not lower it. It
  punishes a relevant chunk that is *late*. This is also why it cannot see a score-gap collapse.
- **The `answer_relevance` threshold of 0.2 is tuned to the fake judge.** Re-tune judged thresholds for a
  real judge.

## Hook 1 statistics

Per query, from the returned scores (sorted high to low): mean, population standard deviation, min, max.

- **score gap** = `scores[0] - scores[min(k, n) - 1]`: the top score minus the k-th (or last, if fewer than k).
- **empty** = no chunks came back.

The window's mean gap only counts queries that returned two or more chunks (a gap needs two scores); if
none did, it is `None` and there is no gap alert. `mean_fill` is the mean of `count / k`.

Per window (the last `window` events for a tenant and route): empty-result rate, nearest-rank
percentiles of latency (`p = sorted[ceil(p/100 * n) - 1]`), and the mean score gap of non-empty queries.

Worked example: scores `[0.9, 0.5, 0.1, 0.05]` with `k = 3` give a gap of `0.9 - 0.1 = 0.8`.

```python
from rag_quality_monitor.hook1 import retrieval_stats
from rag_quality_monitor.hook1.monitor import percentile
from rag_quality_monitor.models import Chunk, RetrievalResult

chunks = [Chunk(id=f"c{i}", doc_id="d", text="t", score=s) for i, s in enumerate([0.9, 0.5, 0.1, 0.05])]
stats = retrieval_stats(RetrievalResult(query="q", chunks=chunks, k=3, latency_ms=12))
assert round(stats.score_gap, 3) == 0.8 and round(stats.mean, 4) == 0.3875
assert percentile([10] * 18 + [500, 900], 95) == 500  # rank ceil(0.95 * 20) = 19
assert percentile([10] * 18 + [500, 900], 99) == 900
```

## Drift (Hook 2)

For each of faithfulness, answer relevance and context precision, compare the new batch mean with the
mean of the last 10 earlier batches (same tenant, route and config hash).

`relative drop = (baseline - current) / baseline`. Alert when it is above 0.15 (exactly 15% is not an alert). With fewer than 3
baseline batches there is no verdict.

Worked example: baseline 0.90, current 0.72 gives a drop of 0.20, so an alert. Current 0.80 gives
0.111, so no alert.

```python
from rag_quality_monitor.hook2 import check_drift
from rag_quality_monitor.models import BatchRecord


def batch(faithfulness, n):
    return BatchRecord(batch_id=f"b{n}", tenant_id="t", route="r", created_at=n, config_hash="h",
                       scored=10, failed=0, faithfulness_mean=faithfulness)


history = [batch(0.90, n) for n in range(5)]
assert len(check_drift(history, batch(0.72, 99))) == 1
assert check_drift(history, batch(0.80, 99)) == []
assert check_drift(history[:2], batch(0.10, 99)) == []
```

## Suite thresholds and regressions (Hook 3)

Per route, each metric is the mean over its cases.

- Algorithmic: pass when `mean >= threshold`.
- Judged: pass when `mean >= threshold - judged_tolerance`.
- Regression versus the previous report: an algorithmic mean that drops at all, or a judged mean that
  drops by more than `judged_tolerance`, even if it is still above its threshold.
