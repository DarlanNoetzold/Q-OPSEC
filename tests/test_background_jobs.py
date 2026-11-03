from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from common.rabbitmq_jobs import RabbitMQJobQueue
from common.redis_cache import RedisTTLCache


async def _async_items(*items):
    for item in items:
        yield item


class RedisCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_round_trips_json_with_ttl_and_tracks_stats(self):
        redis = AsyncMock()
        redis.get.side_effect = [None, b'{"value":1}']
        redis.scan_iter = lambda **_kwargs: _async_items(b"qopsec:cache:key")
        redis.info.return_value = {"evicted_keys": 2}
        cache = RedisTTLCache(redis)

        self.assertIsNone(await cache.get("key"))
        self.assertEqual({"value": 1}, await cache.get("key"))
        await cache.set("key", {"value": 2}, ttl_seconds=15)
        redis.set.assert_awaited_with("qopsec:cache:key", '{"value":2}', ex=15)
        stats = await cache.stats()
        self.assertEqual(1, stats["entries"])
        self.assertEqual(1, stats["hits"])
        self.assertEqual(1, stats["misses"])
        self.assertEqual(2, stats["evictions"])

    async def test_cache_clear_only_removes_its_namespace(self):
        redis = AsyncMock()
        redis.scan_iter = lambda **_kwargs: _async_items(b"qopsec:cache:a", b"qopsec:cache:b")
        await RedisTTLCache(redis).clear()
        redis.delete.assert_awaited_once_with(b"qopsec:cache:a", b"qopsec:cache:b")


class RabbitMQJobQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_persists_status_then_publishes_durable_json_message(self):
        redis = AsyncMock()
        queue = RabbitMQJobQueue(redis, "amqp://guest:guest@localhost/")
        queue._queue = object()
        queue._channel = AsyncMock()

        saved_job = {}

        async def save(job):
            saved_job.update(job)

        queue._save = save
        job_id = await queue.enqueue("training", {"service": "risk_service"})

        self.assertEqual(job_id, saved_job["job_id"])
        self.assertEqual("queued", saved_job["status"])
        message = queue._channel.default_exchange.publish.await_args.args[0]
        self.assertEqual(job_id, message.message_id)
        self.assertEqual("application/json", message.content_type)
        self.assertEqual(json.loads(message.body)["payload"], {"service": "risk_service"})
        self.assertEqual(2, int(message.delivery_mode))

    async def test_consumer_updates_job_state_when_handler_succeeds(self):
        redis = AsyncMock()
        queue = RabbitMQJobQueue(
            redis,
            "amqp://guest:guest@localhost/",
            handler=AsyncMock(return_value={"accepted": True}),
        )
        statuses = []

        async def update(job_id, status, result=None, error=None):
            statuses.append((job_id, status, result, error))

        queue._update = update

        class Message:
            body = json.dumps({"job_id": "abc"}).encode()

            class Process:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *_args):
                    return False

            def process(self, requeue=True):
                return self.Process()

        await queue._on_message(Message())
        self.assertEqual(["running", "succeeded"], [item[1] for item in statuses])
        self.assertEqual({"accepted": True}, statuses[-1][2])


if __name__ == "__main__":
    unittest.main()
