import unittest
from unittest.mock import AsyncMock, patch

import orchestrator_linux as orchestrator


class ArchitectureStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_live_redis_and_rabbitmq_readiness_without_secrets(self):
        redis = AsyncMock()
        queue = AsyncMock()
        queue.health.return_value = {"connected": True, "queue": "qopsec.jobs"}
        with patch.object(orchestrator, "REDIS_CLIENT", redis), patch.object(orchestrator, "JOB_QUEUE", queue):
            with patch.object(orchestrator, "CACHE", AsyncMock(stats=AsyncMock(return_value={"entries": 1, "hits": 2, "misses": 1, "evictions": 0}))):
                with patch.object(orchestrator, "docker_container_running", return_value=True):
                    status = await orchestrator.architecture_status()
        self.assertTrue(status["redis"]["connected"])
        self.assertTrue(status["rabbitmq"]["connected"])
        self.assertNotIn("password", str(status).lower())

    async def test_reports_redis_failure_as_degraded_status(self):
        redis = AsyncMock()
        redis.ping.side_effect = ConnectionError("down")
        queue = AsyncMock()
        queue.health.return_value = {"connected": False, "queue": "qopsec.jobs"}
        queue.stats.return_value = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        with patch.object(orchestrator, "REDIS_CLIENT", redis), patch.object(orchestrator, "JOB_QUEUE", queue):
            with patch.object(orchestrator, "CACHE", AsyncMock(stats=AsyncMock(return_value={"entries": 0, "hits": 0, "misses": 0, "evictions": 0}))):
                with patch.object(orchestrator, "docker_container_running", return_value=False):
                    status = await orchestrator.architecture_status()
        self.assertFalse(status["redis"]["connected"])
        self.assertFalse(status["rabbitmq"]["connected"])


if __name__ == "__main__":
    unittest.main()
