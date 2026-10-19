"""Hook 2, step 2: fire-and-forget capture of a sample after the answer is sent."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple

from pydantic import ValidationError

from ..models import Chunk, RetrievalResult, Sample
from ..store import Store
from .sampler import Sampler

log = logging.getLogger(__name__)

SUSPECT_OVERLAP = 0.8  # share of word 3-grams that must match for the secondary signal to fire
SHORT_CONTEXT_FACTOR = 1.5  # "short context" = at most this many times as many words as the answer


class RawCheck(NamedTuple):
    chunks: list[Chunk] | None  # None means rejected
    reason: str | None
    suspected: bool  # secondary signal: some context looks like an answer (never a hard reject)


def _trigrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def suspected_intermediate_answer(context: str, answer: str) -> bool:
    """Heuristic only. True when the context is mostly inside the answer, or the answer is mostly inside a
    short context, measured by shared word 3-grams. It cannot see a paraphrase: an intermediate answer
    reworded by a model passes. Provenance (`retrieved=`) is the real guard; this is a tripwire."""
    c, a = _trigrams(context), _trigrams(answer)
    if not c or not a:
        return False
    if len(c & a) / len(c) >= SUSPECT_OVERLAP:
        return True
    short = len(context.split()) <= SHORT_CONTEXT_FACTOR * len(answer.split())
    return short and len(c & a) / len(a) >= SUSPECT_OVERLAP


def check_raw_chunks(
    contexts: list[Any], answer: str, retrieved: RetrievalResult | list[str] | None = None
) -> RawCheck:
    """Decide whether the contexts are raw retrieved chunks.

    The design's most important rule: the context is what the retriever returned, never an intermediate
    model's answer. Three layers, strongest first:
    1. Provenance (needs `retrieved`, the retriever's result or its chunk ids): every context id must be
       one the retriever returned for this request, and given a full result the text must match too.
    2. Shape: a context must be a chunk with a non-empty id.
    3. Text heuristic on every context: `suspected_intermediate_answer`. Flag only, never a reject, because
       a short FAQ chunk can legitimately equal an extractive answer.
    """
    chunks: list[Chunk] = []
    for ctx in contexts:
        try:
            chunks.append(ctx if isinstance(ctx, Chunk) else Chunk.model_validate(ctx))
        except ValidationError:
            return RawCheck(None, "context is not a chunk with an id", False)
    if retrieved is not None:
        returned = (
            {c.id: c.text for c in retrieved.chunks}
            if isinstance(retrieved, RetrievalResult)
            else dict.fromkeys(retrieved)
        )
        for c in chunks:
            if c.id not in returned or (returned[c.id] is not None and returned[c.id] != c.text):
                return RawCheck(None, f"context {c.id!r} was not returned by the retriever", False)
    return RawCheck(chunks, None, any(suspected_intermediate_answer(c.text, answer) for c in chunks))


class SampleCapture:
    """Never raises, never blocks the caller. Persistence runs in a background task on a worker thread.

    Simplification: the "queue" is a cap on in-flight tasks (`max_pending`), and overflow is dropped and
    counted. Samples lost on a crash are acceptable for statistics; use a durable queue if they are not.
    """

    def __init__(
        self,
        store: Store,
        sampler: Sampler,
        max_pending: int = 1000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store, self.sampler, self.max_pending, self.clock = store, sampler, max_pending, clock
        self._tasks: set[asyncio.Task] = set()
        self.counters = dict.fromkeys(
            [
                "captured", "sampled_out", "dropped", "rejected_not_raw_chunks", "suspected_intermediate_answer",
                "provenance_unchecked", "failed", "persisted", "duplicate",
            ],
            0,
        )  # fmt: skip

    def capture(
        self,
        tenant_id: str,
        route: str,
        query: str,
        contexts: list[Any],
        answer: str,
        sample_id: str | None = None,
        retrieved: RetrievalResult | list[str] | None = None,
    ) -> str:
        """Returns what happened: queued, sampled_out, dropped, rejected_not_raw_chunks or error.

        Pass `retrieved` (the retriever's result, or its chunk ids) so provenance can be checked. Without
        it only the shape and the text heuristic run, and `provenance_unchecked` counts the gap.
        """
        try:
            checked = check_raw_chunks(contexts, answer, retrieved)
            if checked.chunks is None:  # checked before sampling so a buggy pipeline shows up even at 2%
                self.counters["rejected_not_raw_chunks"] += 1
                return "rejected_not_raw_chunks"
            self.counters["suspected_intermediate_answer"] += checked.suspected
            self.counters["provenance_unchecked"] += retrieved is None
            sample_id = sample_id or uuid.uuid4().hex
            if not self.sampler.should_sample(sample_id, route):
                self.counters["sampled_out"] += 1
                return "sampled_out"
            if len(self._tasks) >= self.max_pending:
                self.counters["dropped"] += 1
                return "dropped"
            sample = Sample(
                sample_id=sample_id, tenant_id=tenant_id, route=route, query=query, chunks=checked.chunks,
                answer=answer, created_at=self.clock(),
            )  # fmt: skip
            loop = asyncio.get_running_loop()  # raises (and is caught below) when there is no loop
            task = loop.create_task(self._persist(sample))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            self.counters["captured"] += 1
            return "queued"
        except Exception:
            log.exception("hook2: capture failed")
            self.counters["failed"] += 1
            return "error"

    async def _persist(self, sample: Sample) -> None:
        try:
            inserted = await asyncio.to_thread(self.store.add_sample, sample)
            self.counters["persisted" if inserted else "duplicate"] += 1
        except Exception:
            log.exception("hook2: could not persist sample")
            self.counters["failed"] += 1

    async def drain(self) -> None:
        """Wait for in-flight captures. For tests, demos and graceful shutdown."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
