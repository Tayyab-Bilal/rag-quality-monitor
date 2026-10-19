"""Pick a judge from the environment. The fake is the default so nothing calls a provider by accident."""

from __future__ import annotations

import os

from .base import Embedder, Judge
from .fake import FakeEmbedder, FakeJudge


def judge_from_env() -> Judge:
    """`RQM_JUDGE=anthropic|openai` plus the matching API key gives a real judge; else the fake."""
    choice = os.environ.get("RQM_JUDGE", "").lower()
    if choice == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        from .anthropic_judge import AnthropicJudge

        return AnthropicJudge(model=os.environ.get("RQM_JUDGE_MODEL", "claude-opus-5-5"))
    if choice == "openai" and os.environ.get("OPENAI_API_KEY"):
        from .openai_judge import OpenAIJudge

        return OpenAIJudge(model=os.environ.get("RQM_JUDGE_MODEL", "gpt-4o-mini"))
    return FakeJudge()


def embedder_from_env() -> Embedder:
    """OpenAI embeddings when the OpenAI judge is selected; the fake otherwise (Anthropic has none)."""
    if os.environ.get("RQM_JUDGE", "").lower() == "openai" and os.environ.get("OPENAI_API_KEY"):
        from .openai_judge import OpenAIEmbedder

        return OpenAIEmbedder()
    return FakeEmbedder()
