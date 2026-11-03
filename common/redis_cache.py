from __future__ import annotations

import json
import time
from typing import Any


class RedisTTLCache:
    """Async Redis-backed TTL cache implementing the orchestrator cache adapter."""

    def __init__(self, redis_client: Any, key_prefix: str = "qopsec:cache:"):
        self._redis = redis_client
        self._prefix = key_prefix
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._last_eviction_check = 0.0

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        value = await self._redis.get(self._key(key))
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    async def set(self, key: str, value: Any, ttl_seconds: float = 30.0) -> None:
        ttl = int(ttl_seconds)
        if ttl <= 0:
            await self.delete(key)
            return
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return await self._redis.set(self._key(key), encoded, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(self._key(key))

    async def clear(self) -> None:
        keys = [key async for key in self._redis.scan_iter(match=f"{self._prefix}*")]
        if keys:
            await self._redis.delete(*keys)

    async def stats(self) -> dict[str, int]:
        keys = [key async for key in self._redis.scan_iter(match=f"{self._prefix}*")]
        now = time.monotonic()
        if now - self._last_eviction_check >= 30:
            try:
                info = await self._redis.info("stats")
                self._evictions = int(info.get("evicted_keys", self._evictions))
            except Exception:
                pass
            self._last_eviction_check = now
        return {
            "entries": len(keys),
            "hits": self._hits,
            "misses": self._misses,
            "max_entries": 0,
            "evictions": self._evictions,
        }
