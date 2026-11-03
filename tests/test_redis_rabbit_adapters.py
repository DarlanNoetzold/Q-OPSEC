from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock

from common.redis_cache import RedisTTLCache
from common.rabbitmq_jobs import RabbitMQJobQueue


class RedisCacheAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_redis_ttl_json_and_namespaced_keys(self):
        redis = AsyncMock()
        redis.get.return_value = b'{"items":[1]}'
        cache = RedisTTLCache(redis_client=redis, key_prefix="test:qopsec:")

        value = await cache.get("dataset-list")
        await cache.set("dataset-list", {"items": [2]}, ttl_seconds=15)

        self.assertEqual({"items": [1]}, value)
        redis.get.assert_awaited_once_with("test:qopsec:dataset-list")
        redis.set.assert_awaited_once_with(
            "test:qopsec:dataset-list", '{"items":[2]}', ex=15
        )

    async def test_stats_report_hits_misses_entries_and_evictions(self):
        redis = AsyncMock()
        redis.scan_iter = lambda **_kwargs: _async_items([b"qopsec:cache:a", b"qopsec:cache:b"])
        redis.info.return_value = {"evicted_keys": 3}
        redis.get.side_effect = [None, b"1", json.dumps({"status": "queued"})]
        cache = RedisTTLCache(redis_client=redis)
        await cache.get("miss")
        await cache.get("hit")

        stats = await cache.stats()
        self.assertEqual(1, stats["hits"])
        self.assertEqual(1, stats["misses"])
        self.assertEqual(2, stats["entries"])
        self.assertEqual(3, stats["evictions"])

    async def test_stats_refreshes_eviction_counter_after_its_cache_interval(self):
        redis = AsyncMock()
        redis.scan_iter = lambda **_kwargs: _async_items([])
        redis.info.return_value = {"evicted_keys": 4}
        cache = RedisTTLCache(redis_client=redis)
        cache._last_eviction_check = time.monotonic() - 31

        stats = await cache.stats()

        self.assertEqual(4, stats["evictions"])
        redis.info.assert_awaited_once_with("stats")


class RabbitQueueAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_publishes_durable_persistent_job_and_tracks_status_in_redis(self):
        connection = AsyncMock()
        connection.is_closed = False
        channel = AsyncMock()
        exchange = AsyncMock()
        queue = AsyncMock()
        channel.default_exchange = exchange
        channel.declare_queue.return_value = queue
        connection.channel.return_value = channel
        redis = AsyncMock()
        adapter = RabbitMQJobQueue(
            redis_client=redis,
            url="amqp://test:test@rabbit.test/",
            queue_name="qopsec.test.jobs",
            connection_factory=AsyncMock(return_value=connection),
        )

        await adapter.start()
        job_id = await adapter.enqueue("training", {"service": "classification_agent"})

        self.assertTrue(job_id)
        redis.set.assert_awaited_once()
        exchange.publish.assert_awaited_once()
        message = exchange.publish.await_args.args[0]
        self.assertEqual(2, message.delivery_mode.value)
        self.assertTrue(message.body)
        self.assertEqual("qopsec.test.jobs", exchange.publish.await_args.kwargs["routing_key"])
        queue.consume.assert_not_awaited()
        await adapter.stop()

    async def test_consumer_marks_job_succeeded_when_handler_returns(self):
        redis = AsyncMock()
        original_job = {
            "job_id": "job1", "job_type": "training", "payload": {"service": "risk_service"},
            "status": "queued", "result": None, "error": None, "attempts": 0,
            "created_at": 1, "updated_at": 1,
        }
        state = {"qopsec:jobs:job1": json.dumps(original_job)}

        async def fake_get(key):
            return state.get(key)

        async def fake_set(key, value, ex=None):
            state[key] = value

        redis.get.side_effect = fake_get
        redis.set.side_effect = fake_set

        async def handler(_job):
            return {"exit_code": 0}

        adapter = RabbitMQJobQueue(redis, "amqp://test", handler=handler)

        class MessageContext:
            def __init__(self, message):
                self.message = message

            async def __aenter__(self):
                return self.message

            async def __aexit__(self, *_exc):
                return False

        class Message:
            body = json.dumps(original_job).encode()

            def process(self, requeue=True):
                return MessageContext(self)

        await adapter._on_message(Message())
        result = json.loads(state["qopsec:jobs:job1"])
        self.assertEqual("succeeded", result["status"])
        self.assertEqual({"exit_code": 0}, result["result"])



def _async_items(items):
    async def iterate(**_kwargs):
        for item in items:
            yield item

    return iterate()


if __name__ == "__main__":
    unittest.main()
