from rag_quality_monitor.judge import FakeEmbedder, FakeJudge
from rag_quality_monitor.metrics.generation import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)

from .conftest import chunk

GOOD = chunk("c1", "Audit logs are retained for 365 days. Older log entries are removed automatically.")
OTHER = chunk("c2", "Sessions expire after 12 hours of inactivity.")


async def test_faithfulness_keeps_per_claim_verdicts_and_unsupported_claims():
    answer = "Audit logs are retained for 365 days. Every plan includes unlimited phone support."
    r = await faithfulness(answer, [GOOD, OTHER], FakeJudge())
    assert r.score == 0.5
    assert [v.supported for v in r.verdicts] == [True, False]
    assert r.verdicts[0].evidence_chunk_id == "c1"
    assert r.verdicts[1].evidence_chunk_id is None
    assert r.unsupported_claims == ["Every plan includes unlimited phone support."]


async def test_faithfulness_with_no_claims_has_no_score():
    r = await faithfulness("   ", [GOOD], FakeJudge())
    assert r.score is None and r.reason


async def test_faithfulness_is_one_when_everything_is_supported():
    r = await faithfulness("Audit logs are retained for 365 days.", [GOOD], FakeJudge())
    assert r.score == 1.0 and r.unsupported_claims == []


async def test_answer_relevance_high_for_on_topic_and_low_for_off_topic():
    q = "How long are audit logs retained?"
    on = await answer_relevance(q, "Audit logs are retained for 365 days.", FakeJudge(), FakeEmbedder())
    off = await answer_relevance(q, "Guests cannot export documents or create workflows.", FakeJudge(), FakeEmbedder())
    assert on > 0.4
    assert off < 0.1
    assert on > off


async def test_context_precision_is_rank_aware():
    q = "How long are audit logs retained?"
    first = await context_precision(q, [GOOD, OTHER, OTHER], FakeJudge())
    last = await context_precision(q, [OTHER, OTHER, GOOD], FakeJudge())
    assert first == 1.0
    assert last < first
    assert abs(last - 1 / 3) < 1e-9


async def test_context_precision_with_nothing_relevant_is_zero():
    assert await context_precision("audit logs retained", [OTHER], FakeJudge()) == 0.0


async def test_context_recall_counts_reference_claims_supported_by_any_chunk():
    ref = "Audit logs are retained for 365 days. Sessions expire after 12 hours of inactivity. Pets are allowed."
    assert await context_recall(ref, [GOOD, OTHER], FakeJudge()) == 2 / 3
    assert await context_recall(ref, [], FakeJudge()) == 0.0


class NoQuestions(FakeJudge):
    async def generate_questions(self, answer, n):
        return []


async def test_answer_relevance_is_none_not_zero_when_no_questions_are_generated():
    assert await answer_relevance("q?", "An answer.", NoQuestions(), FakeEmbedder()) is None
