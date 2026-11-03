from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable
from uuid import uuid4

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message


class RabbitMQJobQueue:
    """Durable RabbitMQ job adapter with Redis-backed job status records."""

    def __init__(
        self,
        redis_client: Any,
        url: str,
        queue_name: str = "qopsec.jobs",
        handler: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
        connection_factory: Callable[..., Awaitable[Any]] | None = None,
        prefetch_count: int = 1,
        job_ttl_seconds: int = 604800,
    ):
        self._redis = redis_client
        self._url = url
        self._queue_name = queue_name
        self._handler = handler
        self._connection_factory = connection_factory or aio_pika.connect_robust
        self._prefetch_count = prefetch_count
        self._job_ttl_seconds = job_ttl_seconds
        self._connection = None
        self._channel = None
        self._queue = None
        self._consumer_tag = None
        self._consumer_task: asyncio.Task | None = None
        self._consumer_stopping = False
        self._local_counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}

    async def start(self) -> None:
        if self._connection is not None and not self._connection.is_closed:
            return
        self._connection = await self._connection_factory(self._url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._prefetch_count)
        self._queue = await self._channel.declare_queue(self._queue_name, durable=True)
        if self._handler is not None:
            self._consumer_stopping = False
            self._consumer_tag = await self._queue.consume(self._on_message, no_ack=False)

    async def stop(self) -> None:
        self._consumer_stopping = True
        if self._queue is not None and self._consumer_tag is not None:
            await self._queue.cancel(self._consumer_tag)
            self._consumer_tag = None
        if self._consumer_task is not None:
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()
        self._connection = self._channel = self._queue = None

    async def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        if self._queue is None:
            raise RuntimeError("RabbitMQ job queue is not connected")
        job_id = uuid4().hex
        now = time.time()
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "payload": payload,
            "status": "queued",
            "result": None,
            "error": None,
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
        }
        await self._save(job)
        message = Message(
            json.dumps(job, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json",
            message_id=job_id,
        )
        try:
            await self._channel.default_exchange.publish(message, routing_key=self._queue_name)
        except Exception:
            await self._update(job_id, "failed", error="Failed to publish job to RabbitMQ")
            raise
        self._local_counts["queued"] += 1
        return job_id

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._status_key(job_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def stats(self) -> dict[str, int]:
        keys = [key async for key in self._redis.scan_iter(match="qopsec:jobs:*")]
        counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        for key in keys:
            raw = await self._redis.get(key)
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            status = json.loads(raw).get("status")
            if status in counts:
                counts[status] += 1
        if self._queue is not None:
            try:
                counts["queued"] = max(counts["queued"], (await self._queue.declaration_result).message_count)
            except Exception:
                pass
        return counts

    async def _save(self, job: dict[str, Any]) -> None:
        await self._redis.set(
            self._status_key(job["job_id"]),
            json.dumps(job, separators=(",", ":"), ensure_ascii=False),
            ex=self._job_ttl_seconds,
        )

    @staticmethod
    def _status_key(job_id: str) -> str:
        return f"qopsec:jobs:{job_id}"

    async def _update(self, job_id: str, status: str, result: Any = None, error: str | None = None) -> None:
        job = await self.get(job_id)
        if job is None:
            return
        job.update(status=status, result=result, error=error, updated_at=time.time())
        await self._save(job)

    async def _on_message(self, message: Any) -> None:
        async with message.process(requeue=True):
            job = json.loads(message.body)
            job_id = job["job_id"]
            if self._handler is None:
                raise RuntimeError("No RabbitMQ job handler is configured")
            await self._update(job_id, "running")
            try:
                result = await self._handler(job)
            except Exception as exc:
                await self._update(job_id, "failed", error=str(exc)[:2000])
                raise
            await self._update(job_id, "succeeded", result=result)

    async def health(self) -> dict[str, Any]:
        return {
            "connected": self._connection is not None and not self._connection.is_closed,
            "queue": self._queue_name,
        }
