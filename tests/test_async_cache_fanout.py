from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import orchestrator_linux as orchestrator


class CachedProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_cache_misses_share_one_upstream_request(self):
        cache = AsyncMock()
        cache.get.side_effect = [None, None]
        cache.set = AsyncMock()

        async def upstream(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return {"value": 1}

        with patch.object(orchestrator, "CACHE", cache), patch.object(
            orchestrator, "proxy_get", new=AsyncMock(side_effect=upstream)
        ) as proxy:
            values = await asyncio.gather(
                orchestrator.cached_proxy_get("http://service.test/data", ttl=5),
                orchestrator.cached_proxy_get("http://service.test/data", ttl=5),
            )

        self.assertEqual([{"value": 1}, {"value": 1}], values)
        proxy.assert_awaited_once()

    async def test_cancelled_request_does_not_cancel_shared_upstream_fetch(self):
        cache = AsyncMock()
        cache.get.side_effect = [None, None]
        cache.set = AsyncMock()
        started = asyncio.Event()
        release = asyncio.Event()

        async def upstream(*_args, **_kwargs):
            started.set()
            await release.wait()
            return {"value": 7}

        with patch.object(orchestrator, "CACHE", cache), patch.object(
            orchestrator, "proxy_get", new=AsyncMock(side_effect=upstream)
        ) as proxy:
            first = asyncio.create_task(orchestrator.cached_proxy_get("http://service.test/cancel", ttl=5))
            await started.wait()
            second = asyncio.create_task(orchestrator.cached_proxy_get("http://service.test/cancel", ttl=5))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual({"value": 7}, await second)

        proxy.assert_awaited_once()

    async def test_cached_proxy_reuses_read_response_with_configured_ttl(self):
        cache = AsyncMock()
        cache.get.return_value = None
        cache.set = AsyncMock()
        expected = {"sessions": [{"id": "s1"}]}
        with patch.object(orchestrator, "CACHE", cache), patch.object(
            orchestrator, "proxy_get", new=AsyncMock(return_value=expected)
        ) as proxy:
            first = await orchestrator.cached_proxy_get(
                "http://service.test/metrics", headers={"X-API-Key": "secret"}, ttl=12
            )
            cache.get.return_value = expected
            second = await orchestrator.cached_proxy_get(
                "http://service.test/metrics", headers={"X-API-Key": "secret"}, ttl=12
            )

        self.assertEqual(expected, first)
        self.assertEqual(expected, second)
        proxy.assert_awaited_once_with(
            "http://service.test/metrics", headers={"X-API-Key": "secret"}
        )
        cache.set.assert_awaited_once()
        self.assertEqual(12, cache.set.await_args.kwargs["ttl_seconds"])

    async def test_cache_key_is_scoped_to_authentication_identity(self):
        first = orchestrator._metrics_cache_key(
            "http://service.test/metrics", {"X-API-Key": "first-secret"}
        )
        second = orchestrator._metrics_cache_key(
            "http://service.test/metrics", {"X-API-Key": "second-secret"}
        )
        self.assertNotEqual(first, second)
        self.assertNotIn("secret", first)
        self.assertNotIn("secret", second)


class ConcurrentLogSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_independent_log_files_and_preserves_service_results(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.log"
            second = Path(directory) / "second.log"
            first.write_text("request-7 first\n", encoding="utf-8")
            second.write_text("request-7 second\n", encoding="utf-8")

            results = await orchestrator.search_logs_concurrently(
                [
                    ("first", str(first), "request-7", 0),
                    ("second", str(second), "request-7", 0),
                ]
            )

        self.assertEqual("request-7 first", results["first"][0]["line"])
        self.assertEqual("request-7 second", results["second"][0]["line"])


if __name__ == "__main__":
    unittest.main()
