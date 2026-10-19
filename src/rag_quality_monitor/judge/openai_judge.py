"""OpenAI judge and embedder. Need the `openai` extra; never imported by the tests."""

from __future__ import annotations

from typing import Any

from .base import JudgeError
from .llm import LLMJudge

_UNSET: Any = object()
_NO_TEMPERATURE = ("o1", "o3", "o4", "gpt-5")  # reasoning models reject a temperature parameter


class OpenAIJudge(LLMJudge):
    """`temperature`: 0 by default, so scores are as repeatable as the API allows. Reasoning models
    (o-series, gpt-5) reject the parameter, so it defaults to None (not sent) for those. Pass a value or
    None to override."""

    def __init__(self, model: str = "gpt-4o-mini", client: Any = None, temperature: float | None = _UNSET) -> None:
        if client is None:
            import openai

            client = openai.AsyncOpenAI()
        self.model = model
        self.client = client
        if temperature is _UNSET:
            temperature = None if model.startswith(_NO_TEMPERATURE) else 0.0
        self.temperature = temperature

    async def _complete(self, system: str, user: str) -> str:
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        resp = await self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **params,
        )
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise JudgeError("the judge model refused to answer")
        if choice.finish_reason == "content_filter":
            raise JudgeError("the judge model's reply was filtered")
        if not choice.message.content:
            raise JudgeError("the judge model returned no text")
        return choice.message.content


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", client: Any = None) -> None:
        if client is None:
            import openai

            client = openai.AsyncOpenAI()
        self.model = model
        self.client = client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
