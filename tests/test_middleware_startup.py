from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock

import orchestrator_linux as orchestrator


class RuntimeDependencyStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_starts_dependencies_before_clients_and_tracks_them(self):
        original_config = orchestrator.CONFIG
        original_cache = orchestrator.CACHE
        original_redis = orchestrator.REDIS_CLIENT
        original_queue = orchestrator.JOB_QUEUE
        original_task = orchestrator._METRICS_TASK
        original_env = {key: os.environ.get(key) for key in (
            "QOPSEC_REDIS_PASSWORD", "QOPSEC_RABBITMQ_USER", "QOPSEC_RABBITMQ_PASSWORD"
        )}
        events = []
        config = {
            "paths": {"logs_dir": "logs"},
            "services": {
                name: {
                    "type": "docker", "container_name": name.replace("_", "-"),
                    "image": "local/test", "labels": ["com.qopsec.managed-by=orchestrator"],
                    "env": {}, "volumes": [], "ports": [], "restart": "unless-stopped",
                }
                for name in ("qopsec_redis", "qopsec_rabbitmq", "qopsec_prometheus", "qopsec_grafana")
            },
        }
        orchestrator.CONFIG = config
        redis = AsyncMock()

        async def record_ping():
            events.append("redis-ping")
            return True

        redis.ping.side_effect = record_ping
        redis.aclose = AsyncMock()
        queue = AsyncMock()
        queue.start.side_effect = lambda: events.append("rabbit-start")
        queue.stop = AsyncMock()
        old_from_url = orchestrator.Redis.from_url
        old_queue_class = orchestrator.RabbitMQJobQueue
        old_start_container = orchestrator.docker_start_container
        old_exists = orchestrator.docker_container_exists
        old_running = orchestrator.docker_container_running
        old_id = orchestrator.docker_get_container_id
        old_labels = orchestrator.docker_container_labels
        old_metrics_loop = orchestrator._metrics_refresh_loop
        old_ready = orchestrator._dependency_ready

        async def start_container(cfg):
            events.append("docker:" + cfg["container_name"])
            return {"status": "started"}

        async def ready(name):
            events.append("ready:" + name)
            return True

        try:
            orchestrator.Redis.from_url = lambda *_args, **_kwargs: redis
            orchestrator.RabbitMQJobQueue = lambda **_kwargs: queue
            orchestrator.docker_start_container = start_container
            orchestrator.docker_container_exists = lambda _name: False
            orchestrator.docker_container_running = lambda _name: True
            orchestrator.docker_get_container_id = lambda _name: "fake-container-id"
            orchestrator.docker_container_labels = lambda _name: []
            orchestrator._dependency_ready = ready
            orchestrator._metrics_refresh_loop = lambda: asyncio.sleep(3600)
            os.environ["QOPSEC_REDIS_PASSWORD"] = "test-only"
            os.environ["QOPSEC_RABBITMQ_USER"] = "test"
            os.environ["QOPSEC_RABBITMQ_PASSWORD"] = "test-only"

            await orchestrator._startup()

            self.assertLess(events.index("docker:qopsec-redis"), events.index("ready:qopsec_redis"))
            self.assertLess(events.index("docker:qopsec-rabbitmq"), events.index("ready:qopsec_rabbitmq"))
            self.assertTrue(orchestrator.STATE["qopsec_redis"]["running"])
            self.assertTrue(orchestrator.STATE["qopsec_rabbitmq"]["running"])
        finally:
            if orchestrator._METRICS_TASK:
                orchestrator._METRICS_TASK.cancel()
                try:
                    await orchestrator._METRICS_TASK
                except asyncio.CancelledError:
                    pass
            orchestrator.CONFIG = original_config
            orchestrator.CACHE = original_cache
            orchestrator.REDIS_CLIENT = original_redis
            orchestrator.JOB_QUEUE = original_queue
            orchestrator._METRICS_TASK = original_task
            orchestrator.Redis.from_url = old_from_url
            orchestrator.RabbitMQJobQueue = old_queue_class
            orchestrator.docker_start_container = old_start_container
            orchestrator.docker_container_exists = old_exists
            orchestrator.docker_container_running = old_running
            orchestrator.docker_get_container_id = old_id
            orchestrator.docker_container_labels = old_labels
            orchestrator._metrics_refresh_loop = old_metrics_loop
            orchestrator._dependency_ready = old_ready
            orchestrator.STATE.clear()
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
