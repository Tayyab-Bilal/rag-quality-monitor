"""Real adapters, exercised with fake clients only. No network."""

from types import SimpleNamespace as NS

import pytest

from rag_quality_monitor.judge import JudgeError, judge_from_env
from rag_quality_monitor.judge.anthropic_judge import AnthropicJudge
from rag_quality_monitor.judge.fake import FakeJudge
from rag_quality_monitor.judge.llm import PROMPT_VERSION, SYSTEM_PROMPT, _Relevant, parse_json
from rag_quality_monitor.judge.openai_judge import OpenAIEmbedder, OpenAIJudge

from .conftest import chunk

CHUNKS = [chunk("c1", "Audit logs are kept 365 days."), chunk("c2", "Other.")]


class FakeAnthropic:
    def __init__(self, *replies, stop_reason="end_turn"):
        self.replies, self.stop_reason, self.calls = list(replies), stop_reason, []
        self.messages = NS(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(stop_reason=self.stop_reason, content=[NS(type="text", text=self.replies.pop(0))])


async def test_anthropic_judge_parses_json_and_sends_no_forbidden_params():
    client = FakeAnthropic('{"claims": ["a", " b "]}')
    judge = AnthropicJudge(client=client)
    assert await judge.extract_claims("a. b.") == ["a", "b"]
    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5-5" and sent["system"] == SYSTEM_PROMPT
    assert sent["max_tokens"] == 16000 and sent["output_config"] == {"effort": "low"}
    assert not {"temperature", "thinking", "budget_tokens"} & set(sent)
    assert sent["messages"] == [{"role": "user", "content": sent["messages"][0]["content"]}]  # no assistant prefill


async def test_anthropic_judge_verify_claims_maps_verdicts_and_drops_invented_chunk_ids():
    reply = '{"verdicts": [{"supported": true, "evidence_chunk_id": "c1"}, {"supported": true, "evidence_chunk_id": "zzz"}, {"supported": false}]}'
    judge = AnthropicJudge(client=FakeAnthropic(reply))
    v = await judge.verify_claims(["x", "y", "z"], CHUNKS)
    assert [(x.claim, x.supported, x.evidence_chunk_id) for x in v] == [("x", True, "c1"), ("y", True, None), ("z", False, None)]


async def test_anthropic_judge_wrong_number_of_verdicts_is_a_judge_error():
    judge = AnthropicJudge(client=FakeAnthropic('{"verdicts": []}'))
    with pytest.raises(JudgeError):
        await judge.verify_claims(["x"], CHUNKS)


@pytest.mark.parametrize("bad", ["not json at all", '{"claims": "nope"}', "{}", ""])
async def test_anthropic_judge_bad_json_raises_judge_error(bad):
    judge = AnthropicJudge(client=FakeAnthropic(bad))
    with pytest.raises(JudgeError):
        await judge.extract_claims("x")


async def test_anthropic_judge_handles_a_fenced_json_reply():
    judge = AnthropicJudge(client=FakeAnthropic('```json\n{"relevant": true}\n```'))
    assert await judge.chunk_relevant("q", CHUNKS[0]) is True


async def test_anthropic_judge_refusal_is_a_judge_error():
    judge = AnthropicJudge(client=FakeAnthropic('{"relevant": true}', stop_reason="refusal"))
    with pytest.raises(JudgeError, match="refused"):
        await judge.chunk_relevant("q", CHUNKS[0])


async def test_anthropic_judge_questions_are_capped_at_n():
    judge = AnthropicJudge(client=FakeAnthropic('{"questions": ["a?", "b?", "c?", "d?"]}'))
    assert await judge.generate_questions("ans", 3) == ["a?", "b?", "c?"]


class FakeOpenAI:
    def __init__(self, content, finish_reason="stop", refusal=None):
        self.calls = []
        msg = NS(content=content, refusal=refusal)
        self._resp = NS(choices=[NS(message=msg, finish_reason=finish_reason)])
        self.chat = NS(completions=NS(create=self._create))
        self.embeddings = NS(create=self._embed)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp

    async def _embed(self, **kwargs):
        return NS(data=[NS(embedding=[1.0, 0.0]), NS(embedding=[0.0, 1.0])])


async def test_openai_judge_uses_json_mode_and_temperature_zero():
    client = FakeOpenAI('{"relevant": false}')
    assert await OpenAIJudge(model="m", client=client).chunk_relevant("q", CHUNKS[0]) is False
    sent = client.calls[0]
    assert sent["response_format"] == {"type": "json_object"} and sent["temperature"] == 0 and sent["model"] == "m"


async def test_openai_judge_bad_json_empty_and_filtered_replies_raise_judge_error():
    for client in (FakeOpenAI("nope"), FakeOpenAI(None), FakeOpenAI('{"relevant": true}', "content_filter")):
        with pytest.raises(JudgeError):
            await OpenAIJudge(client=client).chunk_relevant("q", CHUNKS[0])


async def test_openai_embedder_returns_vectors():
    assert await OpenAIEmbedder(client=FakeOpenAI("")).embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_parse_json_validates_the_schema():
    assert parse_json('{"relevant": true}', _Relevant).relevant is True
    with pytest.raises(JudgeError):
        parse_json('{"relevant": "maybe-ish"}', _Relevant)


def test_prompt_version_is_a_constant_string():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_judge_from_env_defaults_to_the_fake_and_needs_a_key_for_a_real_one(monkeypatch):
    for var in ("RQM_JUDGE", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(judge_from_env(), FakeJudge)
    monkeypatch.setenv("RQM_JUDGE", "anthropic")
    assert isinstance(judge_from_env(), FakeJudge)  # selected but no key: stay on the fake


async def test_anthropic_judge_max_tokens_stop_gets_a_clear_judge_error():
    judge = AnthropicJudge(client=FakeAnthropic('{"claims": ["cut', stop_reason="max_tokens"))
    with pytest.raises(JudgeError, match="max_tokens"):
        await judge.extract_claims("x")


async def test_anthropic_judge_ignores_non_text_blocks():
    client = FakeAnthropic("unused")

    async def create(**kwargs):
        blocks = [NS(type="thinking", thinking="hmm"), NS(type="text", text='{"relevant": true}')]
        return NS(stop_reason="end_turn", content=blocks)

    client.messages = NS(create=create)
    assert await AnthropicJudge(client=client).chunk_relevant("q", CHUNKS[0]) is True


async def test_openai_judge_refusal_is_a_judge_error():
    with pytest.raises(JudgeError, match="refused"):
        await OpenAIJudge(client=FakeOpenAI(None, refusal="I can't help with that")).chunk_relevant("q", CHUNKS[0])


async def test_openai_judge_sends_temperature_only_when_the_model_accepts_it():
    plain, reasoning, custom = FakeOpenAI('{"relevant": true}'), FakeOpenAI('{"relevant": true}'), FakeOpenAI('{"relevant": true}')
    await OpenAIJudge(model="gpt-4o-mini", client=plain).chunk_relevant("q", CHUNKS[0])
    await OpenAIJudge(model="o3-mini", client=reasoning).chunk_relevant("q", CHUNKS[0])
    await OpenAIJudge(model="gpt-4o", client=custom, temperature=None).chunk_relevant("q", CHUNKS[0])
    assert plain.calls[0]["temperature"] == 0
    assert "temperature" not in reasoning.calls[0] and "temperature" not in custom.calls[0]
