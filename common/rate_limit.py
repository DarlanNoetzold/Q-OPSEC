"""Distributed rate limiting shared by every replica.

Redis is the source of truth (fixed window counter with INCR + EXPIRE), with an
in-process fallback so a single-replica deployment still enforces a limit when
Redis is unreachable. Keys are hashed identities, never raw API keys.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


@dataclass
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after: float
    backend: str

    def headers(self) -> Dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(max(0.0, self.reset_after))),
            "X-RateLimit-Backend": self.backend,
        }


class RedisRateLimiter:
    backend = "redis"

    def __init__(self, client: Any, prefix: str = "qopsec:ratelimit:", window_seconds: int = 60, limit: int = 600):
        self._client = client
        self._prefix = prefix
        self._window = max(1, int(window_seconds))
        self._limit = max(1, int(limit))

    async def check(self, identity: str, cost: int = 1) -> RateLimitDecision:
        window = int(time.time()) // self._window
        key = f"{self._prefix}{identity}:{window}"
        pipe = self._client.pipeline()
        pipe.incrby(key, cost)
        pipe.expire(key, self._window)
        result = pipe.execute()
        if inspect.isawaitable(result):
            result = await result
        used, _ = result
        used = int(used)
        reset_after = self._window - (time.time() % self._window)
        return RateLimitDecision(
            allowed=used <= self._limit,
            limit=self._limit,
            remaining=max(0, self._limit - used),
            reset_after=reset_after,
            backend=self.backend,
        )


class InMemoryRateLimiter:
    backend = "memory"

    def __init__(self, window_seconds: int = 60, limit: int = 600, clock: Callable[[], float] = time.time):
        self._window = max(1, int(window_seconds))
        self._limit = max(1, int(limit))
        self._clock = clock
        self._counters: Dict[Tuple[str, int], int] = {}

    async def check(self, identity: str, cost: int = 1) -> RateLimitDecision:
        now = self._clock()
        window = int(now) // self._window
        key = (identity, window)
        if len(self._counters) > 4096:
            self._counters = {k: v for k, v in self._counters.items() if k[1] >= window}
        used = self._counters.get(key, 0) + cost
        self._counters[key] = used
        reset_after = self._window - (now % self._window)
        return RateLimitDecision(
            allowed=used <= self._limit,
            limit=self._limit,
            remaining=max(0, self._limit - used),
            reset_after=reset_after,
            backend=self.backend,
        )


def build_rate_limiter(
    redis_client: Any = None,
    window_seconds: int = 60,
    limit: int = 600,
    prefix: str = "qopsec:ratelimit:",
) -> Any:
    if redis_client is not None:
        return RedisRateLimiter(redis_client, prefix=prefix, window_seconds=window_seconds, limit=limit)
    return InMemoryRateLimiter(window_seconds=window_seconds, limit=limit)


class FailoverRateLimiter:
    """Uses Redis when the client is available, otherwise the local counter."""

    backend = "redis+memory"

    def __init__(
        self,
        client_getter: Callable[[], Any],
        window_seconds: int = 60,
        limit: int = 600,
        prefix: str = "qopsec:ratelimit:",
    ):
        self._client_getter = client_getter
        self._window = max(1, int(window_seconds))
        self._limit = max(1, int(limit))
        self._prefix = prefix
        self._memory = InMemoryRateLimiter(window_seconds=self._window, limit=self._limit)
        self._redis: Optional[RedisRateLimiter] = None
        self._redis_client_id: Optional[int] = None
        self.last_backend = "memory"

    async def check(self, identity: str, cost: int = 1) -> RateLimitDecision:
        client = self._client_getter()
        if client is not None:
            if self._redis is None or self._redis_client_id != id(client):
                self._redis = RedisRateLimiter(
                    client, prefix=self._prefix, window_seconds=self._window, limit=self._limit
                )
                self._redis_client_id = id(client)
            try:
                decision = await self._redis.check(identity, cost)
                self.last_backend = decision.backend
                return decision
            except Exception:
                self._redis = None
                self._redis_client_id = None
        decision = await self._memory.check(identity, cost)
        self.last_backend = decision.backend
        return decision


def build_failover_limiter(
    client_getter: Callable[[], Any],
    window_seconds: int = 60,
    limit: int = 600,
    prefix: str = "qopsec:ratelimit:",
) -> FailoverRateLimiter:
    return FailoverRateLimiter(client_getter, window_seconds=window_seconds, limit=limit, prefix=prefix)


def identity_for_request(request: Request) -> str:
    """Hash the caller identity so no credential ever lands in a Redis key."""
    api_key = request.headers.get("x-api-key") or request.headers.get("authorization")
    if api_key:
        digest = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return f"key:{digest}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies the limiter to the configured path prefixes."""

    def __init__(
        self,
        app: Any,
        limiter: Any,
        paths: Tuple[str, ...] = ("/run-pipeline",),
        exclude_paths: Tuple[str, ...] = ("/health", "/architecture", "/pipeline/jobs"),
        on_decision: Optional[Callable[[RateLimitDecision], None]] = None,
    ):
        super().__init__(app)
        self._limiter = limiter
        self._paths = tuple(paths)
        self._exclude = tuple(exclude_paths)
        self._on_decision = on_decision

    def _applies(self, path: str) -> bool:
        if any(path.startswith(prefix) for prefix in self._exclude):
            return False
        return any(path.startswith(prefix) for prefix in self._paths)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if not self._applies(request.url.path):
            return await call_next(request)
        decision = await self._limiter.check(identity_for_request(request))
        callback = getattr(self._limiter, "on_decision", None) or self._on_decision
        if callback is not None:
            callback(decision)
        if not decision.allowed:
            return JSONResponse(
                {"detail": "Rate limit exceeded", "limit": decision.limit, "reset_after": decision.reset_after},
                status_code=429,
                headers=decision.headers(),
            )
        response = await call_next(request)
        for key, value in decision.headers().items():
            response.headers[key] = value
        return response
