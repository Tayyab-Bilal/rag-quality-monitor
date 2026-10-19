"""Deterministic judge and embedder for tests, the demo and CI.

The heuristics are crude on purpose; they are documented so a failing test can be reasoned about.
- Claims: the text split into sentences.
- Supported: the share of the claim's content words found in a chunk is >= `support_threshold`.
  Evidence is the best-matching chunk.
- Relevant: the share of the question's content words found in the chunk is >= `relevance_threshold`.
- Generated questions: the content words of sentence i (up to 8), as a question. No filler words, so the
  cosine with a real question is limited only by how much of the answer the question covers.
"""

from __future__ import annotations

import hashlib
import math
import re

from ..models import Chunk, ClaimVerdict

STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be by it its this that these those as at "
    "from can you your how what which when do does did will would should".split()
)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def content_words(text: str) -> set[str]:
    """Lower-cased words minus stopwords and one-letter noise, with a crude plural stem."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


def overlap(needle: str, haystack: str) -> float:
    """Share of the needle's content words that also appear in the haystack."""
    n = content_words(needle)
    return len(n & content_words(haystack)) / len(n) if n else 0.0


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


class FakeJudge:
    def __init__(
        self,
        support_threshold: float = 0.6,
        relevance_threshold: float = 0.6,
        model: str = "fake-judge-v1",
    ) -> None:
        self.support_threshold = support_threshold
        self.relevance_threshold = relevance_threshold
        self.model = model

    async def extract_claims(self, text: str) -> list[str]:
        return split_sentences(text)

    async def verify_claims(self, claims: list[str], chunks: list[Chunk]) -> list[ClaimVerdict]:
        verdicts = []
        for claim in claims:
            best_id, best = None, 0.0
            for chunk in chunks:
                score = overlap(claim, chunk.text)
                if score > best:
                    best_id, best = chunk.id, score
            supported = best >= self.support_threshold
            verdicts.append(
                ClaimVerdict(claim=claim, supported=supported, evidence_chunk_id=best_id if supported else None)
            )
        return verdicts

    async def chunk_relevant(self, question: str, chunk: Chunk) -> bool:
        return overlap(question, chunk.text) >= self.relevance_threshold

    async def generate_questions(self, answer: str, n: int) -> list[str]:
        sentences = split_sentences(answer)
        if not sentences:
            return []
        out = []
        for i in range(n):
            words = sorted(content_words(sentences[i % len(sentences)]))[:8]
            out.append(f"{' '.join(words)}?")
        return out


class FakeEmbedder:
    """Hashed bag of content words, L2-normalised. Same words in, same vector out, no model."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model = f"fake-hash-{dim}"  # identity, part of the batch config hash

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in content_words(text):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out
