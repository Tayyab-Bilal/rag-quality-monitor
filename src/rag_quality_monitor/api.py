"""FastAPI surface. Needs the `api` extra. The tenant always comes from the verified token."""

from __future__ import annotations

from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .hook1.monitor import RetrievalMonitor
from .hook2.batch import run_batch
from .judge.base import Embedder, Judge
from .store import Store


class TokenVerifier(Protocol):
    def verify(self, token: str) -> str | None:
        """Return the tenant id for a valid token, else None."""


class FakeTokenVerifier:
    def __init__(self, tokens: dict[str, str]) -> None:
        self.tokens = tokens

    def verify(self, token: str) -> str | None:
        return self.tokens.get(token)


class BatchRequest(BaseModel):
    route: str
    min_samples: int = 10  # same default as run_batch
    max_judge_calls: int | None = None  # may only lower the server's cap, never raise it


def create_app(
    store: Store, judge: Judge, embedder: Embedder, verifier: TokenVerifier, max_judge_calls: int = 500
) -> FastAPI:
    """`max_judge_calls` is server-side policy: a caller must not be able to set its own spending limit."""
    app = FastAPI(title="rag-quality-monitor")
    monitor = RetrievalMonitor(store)

    def tenant(authorization: str | None = Header(default=None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        found = verifier.verify(authorization.removeprefix("Bearer "))
        if found is None:
            raise HTTPException(401, "invalid token")
        return found

    @app.get("/stats")
    def stats(route: str, tenant_id: str = Depends(tenant)) -> dict:
        window = monitor.window_stats(tenant_id, route)
        batch = store.latest_batch(tenant_id, route)
        return {
            "route": route,
            "hook1": window.model_dump() if window else None,
            "hook2": batch.model_dump() if batch else None,
            "drift": [a.model_dump() for a in batch.drift] if batch else [],
        }

    @app.get("/samples/{sample_id}")
    def sample(sample_id: str, tenant_id: str = Depends(tenant)) -> dict:
        found = store.get_sample(tenant_id, sample_id)
        if found is None:  # same answer for "not yours" and "does not exist"
            raise HTTPException(404, "sample not found")
        return found

    @app.post("/batches/run")
    async def batches_run(body: BatchRequest, tenant_id: str = Depends(tenant)) -> dict:
        result = await run_batch(
            store, judge, embedder, tenant_id, body.route,
            min_samples=body.min_samples,
            max_judge_calls=min(body.max_judge_calls, max_judge_calls) if body.max_judge_calls is not None else max_judge_calls,
        )  # fmt: skip
        return result.model_dump()

    @app.get("/golden/report")
    def golden_report(tenant_id: str = Depends(tenant)) -> dict:
        raw = store.latest_suite_report(tenant_id)
        if raw is None:
            raise HTTPException(404, "no suite report saved")
        import json

        return json.loads(raw)

    return app
