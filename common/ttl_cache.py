from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Any, Callable


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

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl_seconds: float = 30.0) -> Any:
        """Return the cached value, computing and storing it on a miss."""
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.set(key, value, ttl_seconds)
        return value

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
