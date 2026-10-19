"""The seams: a Judge and an Embedder are Protocols so tests never need a provider."""

from __future__ import annotations

from typing import Protocol

from ..models import Chunk, ClaimVerdict


class JudgeError(Exception):
    """The judge could not give a usable answer (bad JSON, refusal). Callers record eval_failed."""


class Judge(Protocol):
    model: str

    async def extract_claims(self, text: str) -> list[str]: ...

    async def verify_claims(self, claims: list[str], chunks: list[Chunk]) -> list[ClaimVerdict]: ...

    async def chunk_relevant(self, question: str, chunk: Chunk) -> bool: ...

    async def generate_questions(self, answer: str, n: int) -> list[str]: ...


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class CountingJudge:
    """Wraps any judge and counts calls. One wrapper per batch keeps the count per batch even when
    several batches share one judge, which is what a cost cap needs."""

    def __init__(self, judge: Judge) -> None:
        self._judge = judge
        self.model = judge.model
        self.calls = 0

    async def extract_claims(self, text: str) -> list[str]:
        self.calls += 1
        return await self._judge.extract_claims(text)

    async def verify_claims(self, claims: list[str], chunks: list[Chunk]) -> list[ClaimVerdict]:
        self.calls += 1
        return await self._judge.verify_claims(claims, chunks)

    async def chunk_relevant(self, question: str, chunk: Chunk) -> bool:
        self.calls += 1
        return await self._judge.chunk_relevant(question, chunk)

    async def generate_questions(self, answer: str, n: int) -> list[str]:
        self.calls += 1
        return await self._judge.generate_questions(answer, n)
