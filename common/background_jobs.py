from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Awaitable, Callable
from uuid import uuid4


class TTLCache:
    """Small bounded in-process TTL cache for non-sensitive read-only data."""

    def __init__(self, max_entries: int = 256, clock: Callable[[], float] = time.monotonic):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return deepcopy(value)

    def set(self, key: str, value: Any, ttl_seconds: float = 30.0) -> None:
        if ttl_seconds <= 0:
            self.delete(key)
            return
        expires_at = self._clock() + ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, deepcopy(value))
            self._entries.move_to_end(key)
            self._evict_expired_locked(self._clock())
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._evict_expired_locked(self._clock())
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "max_entries": self._max_entries,
            }

    def _evict_expired_locked(self, now: float) -> None:
        expired = [key for key, (expiry, _) in self._entries.items() if expiry <= now]
        for key in expired:
            self._entries.pop(key, None)


class SQLiteJobRepository:
    """Durable single-host job store. Each call owns a short-lived SQLite connection."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS background_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
                    result TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_background_jobs_status_created "
                "ON background_jobs(status, created_at)"
            )
            # A worker crash should not strand a job permanently.
            connection.execute(
                "UPDATE background_jobs SET status='queued', updated_at=? WHERE status='running'",
                (time.time(),),
            )

    def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        job_id = uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO background_jobs(job_id,job_type,payload,status,created_at,updated_at) "
                "VALUES(?,?,?,'queued',?,?)",
                (job_id, job_type, json.dumps(payload, separators=(",", ":")), now, now),
            )
        return job_id

    def claim_next(self) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM background_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE background_jobs SET status='running', attempts=attempts+1, updated_at=? "
                "WHERE job_id=? AND status='queued'",
                (now, row["job_id"]),
            )
            claimed = dict(row)
            claimed["status"] = "running"
            claimed["attempts"] += 1
            claimed["updated_at"] = now
            claimed["payload"] = json.loads(claimed["payload"])
            return claimed

    def finish(self, job_id: str, result: Any) -> None:
        self._update(job_id, "succeeded", result=json.dumps(result, default=str), error=None)

    def fail(self, job_id: str, error: str) -> None:
        self._update(job_id, "failed", result=None, error=error[:2000])

    def _update(self, job_id: str, status: str, result: str | None, error: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE background_jobs SET status=?, result=?, error=?, updated_at=? WHERE job_id=?",
                (status, result, error, time.time(), job_id),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM background_jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._decode(dict(row)) if row else None

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM background_jobs GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        return {status: counts.get(status, 0) for status in ("queued", "running", "succeeded", "failed")}

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        row["payload"] = json.loads(row["payload"])
        row["result"] = json.loads(row["result"]) if row["result"] else None
        return row


class BackgroundJobQueue:
    """One local worker loop; state is durable but intentionally single-host."""

    def __init__(
        self,
        repository: SQLiteJobRepository,
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        poll_interval: float = 0.5,
    ):
        self.repository = repository
        self.handler = handler
        self.poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task = asyncio.create_task(self._run(), name="qopsec-job-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping.set()
        await self._task
        self._task = None

    async def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        return await asyncio.to_thread(self.repository.enqueue, job_type, payload)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.repository.get, job_id)

    async def stats(self) -> dict[str, int]:
        return await asyncio.to_thread(self.repository.stats)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            job = await asyncio.to_thread(self.repository.claim_next)
            if job is None:
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                result = await self.handler(job)
            except Exception as exc:
                await asyncio.to_thread(self.repository.fail, job["job_id"], str(exc))
            else:
                await asyncio.to_thread(self.repository.finish, job["job_id"], result)
