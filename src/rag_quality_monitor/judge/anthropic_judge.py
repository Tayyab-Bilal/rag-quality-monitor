"""Claude judge. Needs the `anthropic` extra; never imported by the tests.

Claude Opus 5.5 always thinks, and `effort` is the control for how much. Temperature, thinking budgets and
assistant prefill are rejected, so none of them are sent: the system prompt asks for JSON and `parse_json`
validates it. Thinking tokens count against `max_tokens`, hence the large limit.
"""

from __future__ import annotations

from typing import Any

from .base import JudgeError
from .llm import LLMJudge


class AnthropicJudge(LLMJudge):
    def __init__(self, model: str = "claude-opus-5-5", client: Any = None, max_tokens: int = 16000, effort: str = "low") -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self.model = model
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort

    async def _complete(self, system: str, user: str) -> str:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise JudgeError("the judge model refused to answer")
        if resp.stop_reason == "max_tokens":
            raise JudgeError(f"the judge reply was cut off at max_tokens={self.max_tokens}; raise max_tokens or lower effort")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise JudgeError("the judge model returned no text")
        return text
