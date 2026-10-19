"""One SQLite store for events, samples, scores, batches and suite reports.

Every read and write takes a tenant id and puts it in the WHERE clause: a missing tenant is a
TypeError, not a data leak. Each Store owns one connection (guarded by a lock) so it can be used from
`asyncio.to_thread`; to race several workers, give each its own Store on the same file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from .models import BatchRecord, Chunk, Sample

SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, route TEXT NOT NULL,
  ts REAL NOT NULL, stats TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS samples (
  tenant_id TEXT NOT NULL, sample_id TEXT NOT NULL, route TEXT NOT NULL, query TEXT NOT NULL,
  chunks TEXT NOT NULL, answer TEXT NOT NULL, created_at REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', claim_token TEXT, claimed_at REAL, error TEXT,
  faithfulness REAL, answer_relevance REAL, context_precision REAL,
  verdicts TEXT, attribution TEXT, config_hash TEXT,
  PRIMARY KEY (tenant_id, sample_id));
CREATE TABLE IF NOT EXISTS batches (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, route TEXT NOT NULL,
  config_hash TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS suite_reports (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, data TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def _all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock, self._conn:
            return self._conn.execute(sql, params).fetchall()

    def _write(self, sql: str, params: tuple | dict = ()) -> int:
        with self._lock, self._conn:
            return self._conn.execute(sql, params).rowcount

    # Hook 1 -------------------------------------------------------------------------------
    def add_event(self, tenant_id: str, route: str, ts: float, stats_json: str) -> None:
        self._write(
            "INSERT INTO retrieval_events (tenant_id, route, ts, stats) VALUES (?, ?, ?, ?)",
            (tenant_id, route, ts, stats_json),
        )

    def recent_events(self, tenant_id: str, route: str, limit: int) -> list[tuple[float, str]]:
        rows = self._all(
            "SELECT ts, stats FROM retrieval_events WHERE tenant_id=? AND route=? ORDER BY seq DESC LIMIT ?",
            (tenant_id, route, limit),
        )
        return [(r["ts"], r["stats"]) for r in rows]

    # Hook 2 samples -----------------------------------------------------------------------
    def add_sample(self, sample: Sample) -> bool:
        """True if inserted, False if a sample with this (tenant, id) already existed."""
        return self._write(
            "INSERT OR IGNORE INTO samples (tenant_id, sample_id, route, query, chunks, answer, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                sample.tenant_id,
                sample.sample_id,
                sample.route,
                sample.query,
                json.dumps([c.model_dump() for c in sample.chunks]),
                sample.answer,
                sample.created_at,
            ),
        ) > 0

    def pending_info(self, tenant_id: str, route: str) -> tuple[int, float | None]:
        row = self._all(
            "SELECT COUNT(*) AS n, MIN(created_at) AS oldest FROM samples"
            " WHERE tenant_id=? AND route=? AND status='pending'",
            (tenant_id, route),
        )[0]
        return row["n"], row["oldest"]

    def reclaim_stale(self, tenant_id: str, route: str, older_than: float) -> int:
        """Return samples claimed before `older_than` to pending: their worker died or failed to save."""
        return self._write(
            "UPDATE samples SET status='pending', claim_token=NULL, claimed_at=NULL"
            " WHERE tenant_id=? AND route=? AND status='claimed' AND claimed_at < ?",
            (tenant_id, route, older_than),
        )

    def claim_samples(
        self, tenant_id: str, route: str, limit: int, now: float | None = None
    ) -> tuple[str, list[Sample]]:
        """Atomically move up to `limit` pending samples to 'claimed' under a fresh token.

        One conditional UPDATE: SQLite serialises writers, so two workers can never claim the same row.
        Postgres would use `FOR UPDATE SKIP LOCKED` in the subquery instead.
        """
        token = uuid.uuid4().hex
        self._write(
            "UPDATE samples SET status='claimed', claim_token=:tok, claimed_at=:now WHERE tenant_id=:t AND route=:r"
            " AND status='pending' AND sample_id IN (SELECT sample_id FROM samples WHERE tenant_id=:t"
            " AND route=:r AND status='pending' ORDER BY created_at, sample_id LIMIT :n)",
            {"tok": token, "now": time.time() if now is None else now, "t": tenant_id, "r": route, "n": limit},
        )
        rows = self._all(
            "SELECT * FROM samples WHERE tenant_id=? AND claim_token=? ORDER BY created_at, sample_id",
            (tenant_id, token),
        )
        return token, [self._to_sample(r) for r in rows]

    def release(self, tenant_id: str, token: str) -> int:
        """Give unprocessed claimed samples back to the pending pool (cost cap reached)."""
        return self._write(
            "UPDATE samples SET status='pending', claim_token=NULL, claimed_at=NULL"
            " WHERE tenant_id=? AND claim_token=? AND status='claimed'",
            (tenant_id, token),
        )

    def save_scores(
        self,
        tenant_id: str,
        sample_id: str,
        token: str,
        *,
        faithfulness: float | None,
        answer_relevance: float,
        context_precision: float,
        verdicts: list[dict],
        attribution: list[str] | None,
        config_hash: str,
    ) -> None:
        self._write(
            "UPDATE samples SET status='scored', faithfulness=?, answer_relevance=?, context_precision=?,"
            " verdicts=?, attribution=?, config_hash=? WHERE tenant_id=? AND sample_id=? AND claim_token=?",
            (
                faithfulness,
                answer_relevance,
                context_precision,
                json.dumps(verdicts),
                json.dumps(attribution) if attribution is not None else None,
                config_hash,
                tenant_id,
                sample_id,
                token,
            ),
        )

    def mark_failed(self, tenant_id: str, sample_id: str, token: str, error: str) -> None:
        self._write(
            "UPDATE samples SET status='eval_failed', error=? WHERE tenant_id=? AND sample_id=? AND claim_token=?",
            (error, tenant_id, sample_id, token),
        )

    def get_sample(self, tenant_id: str, sample_id: str) -> dict[str, Any] | None:
        """The drill-down: the raw chunks, the answer, the scores and every claim verdict."""
        rows = self._all("SELECT * FROM samples WHERE tenant_id=? AND sample_id=?", (tenant_id, sample_id))
        if not rows:
            return None
        r = rows[0]
        return {
            "sample_id": r["sample_id"],
            "route": r["route"],
            "query": r["query"],
            "answer": r["answer"],
            "chunks": json.loads(r["chunks"]),
            "status": r["status"],
            "error": r["error"],
            "faithfulness": r["faithfulness"],
            "answer_relevance": r["answer_relevance"],
            "context_precision": r["context_precision"],
            "verdicts": json.loads(r["verdicts"]) if r["verdicts"] else [],
            "unsupported_claims": json.loads(r["attribution"]) if r["attribution"] else [],
            "config_hash": r["config_hash"],
        }

    def count_by_status(self, tenant_id: str, route: str) -> dict[str, int]:
        rows = self._all(
            "SELECT status, COUNT(*) AS n FROM samples WHERE tenant_id=? AND route=? GROUP BY status",
            (tenant_id, route),
        )
        return {r["status"]: r["n"] for r in rows}

    @staticmethod
    def _to_sample(r: sqlite3.Row) -> Sample:
        return Sample(
            sample_id=r["sample_id"],
            tenant_id=r["tenant_id"],
            route=r["route"],
            query=r["query"],
            chunks=[Chunk(**c) for c in json.loads(r["chunks"])],
            answer=r["answer"],
            created_at=r["created_at"],
        )

    # Batches and reports ------------------------------------------------------------------
    def add_batch(self, record: BatchRecord) -> None:
        self._write(
            "INSERT INTO batches (tenant_id, route, config_hash, data) VALUES (?, ?, ?, ?)",
            (record.tenant_id, record.route, record.config_hash, record.model_dump_json()),
        )

    def recent_batches(
        self, tenant_id: str, route: str, limit: int, config_hash: str | None = None
    ) -> list[BatchRecord]:
        """Newest first. Only batches that scored something can serve as a baseline."""
        sql = "SELECT data FROM batches WHERE tenant_id=? AND route=?"
        params: list[Any] = [tenant_id, route]
        if config_hash is not None:
            sql += " AND config_hash=?"
            params.append(config_hash)
        rows = self._all(sql + " ORDER BY seq DESC LIMIT ?", (*params, limit))
        return [BatchRecord.model_validate_json(r["data"]) for r in rows]

    def latest_batch(self, tenant_id: str, route: str) -> BatchRecord | None:
        batches = self.recent_batches(tenant_id, route, 1)
        return batches[0] if batches else None

    def save_suite_report(self, tenant_id: str, report_json: str) -> None:
        self._write("INSERT INTO suite_reports (tenant_id, data) VALUES (?, ?)", (tenant_id, report_json))

    def latest_suite_report(self, tenant_id: str) -> str | None:
        rows = self._all("SELECT data FROM suite_reports WHERE tenant_id=? ORDER BY seq DESC LIMIT 1", (tenant_id,))
        return rows[0]["data"] if rows else None
