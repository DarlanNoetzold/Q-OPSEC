"""Circuit breaker, retry with backoff and a resilient HTTP client.

No third-party dependency: the breaker and the retry loop are plain asyncio, so
every service can import this module without touching its requirements file.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

import httpx


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is open."""


class RetryableStatusError(httpx.HTTPStatusError):
    """Upstream answered with a retryable status (502/503/504 by default)."""


@dataclass
class BreakerSnapshot:
    name: str
    state: str
    failures: int
    successes: int
    rejections: int
    opened_at: Optional[float]
    last_error: Optional[str]
    retry_after_seconds: float


class CircuitBreaker:
    """Failure-threshold breaker with a half-open probe window."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be positive")
        self.name = name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._state = "closed"
        self._failures = 0
        self._successes = 0
        self._rejections = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._last_error: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    def _should_probe(self) -> bool:
        return (
            self._state == "open"
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self._reset_timeout
        )

    async def _acquire(self) -> None:
        async with self._lock:
            if self._should_probe():
                self._state = "half_open"
                self._half_open_calls = 0
            if self._state == "open":
                self._rejections += 1
                retry_after = self._reset_timeout - (self._clock() - (self._opened_at or 0))
                raise CircuitOpenError(
                    f"circuit '{self.name}' is open; retry in {max(0.0, retry_after):.1f}s"
                )
            if self._state == "half_open":
                if self._half_open_calls >= self._half_open_max_calls:
                    self._rejections += 1
                    raise CircuitOpenError(f"circuit '{self.name}' is half-open and busy")
                self._half_open_calls += 1

    async def _record_success(self) -> None:
        async with self._lock:
            self._successes += 1
            self._failures = 0
            self._state = "closed"
            self._opened_at = None
            self._half_open_calls = 0

    async def _record_failure(self, exc: BaseException) -> None:
        async with self._lock:
            self._failures += 1
            self._last_error = f"{type(exc).__name__}: {exc}"[:200]
            if self._state == "half_open" or self._failures >= self._failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()
                self._half_open_calls = 0

    async def call(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        await self._acquire()
        try:
            result = await fn()
        except BaseException as exc:
            await self._record_failure(exc)
            raise
        await self._record_success()
        return result

    def snapshot(self) -> BreakerSnapshot:
        retry_after = 0.0
        if self._state == "open" and self._opened_at is not None:
            retry_after = max(0.0, self._reset_timeout - (self._clock() - self._opened_at))
        return BreakerSnapshot(
            name=self.name,
            state=self._state,
            failures=self._failures,
            successes=self._successes,
            rejections=self._rejections,
            opened_at=self._opened_at,
            last_error=self._last_error,
            retry_after_seconds=round(retry_after, 2),
        )

    def reset(self) -> None:
        self._state = "closed"
        self._failures = 0
        self._opened_at = None
        self._half_open_calls = 0
        self._last_error = None


class BreakerRegistry:
    """Keeps one breaker per dependency name so state survives across calls."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0):
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name,
                failure_threshold=self._failure_threshold,
                reset_timeout=self._reset_timeout,
            )
            self._breakers[name] = breaker
        return breaker

    def snapshots(self) -> list[BreakerSnapshot]:
        return [breaker.snapshot() for breaker in self._breakers.values()]

    def open_breakers(self) -> list[str]:
        return [name for name, breaker in self._breakers.items() if breaker.state == "open"]


@dataclass
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0
    jitter: float = 0.3
    retry_on: Iterable[type] = field(default_factory=lambda: (httpx.TransportError, httpx.TimeoutException))

    def delay_for(self, attempt: int) -> float:
        raw = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        if self.jitter <= 0:
            return raw
        return raw * (1 + random.uniform(-self.jitter, self.jitter))


async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Any:
    """Run ``fn`` retrying transient failures with exponential backoff and jitter."""
    policy = policy or RetryPolicy()
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, policy.attempts) + 1):
        try:
            return await fn()
        except CircuitOpenError:
            raise
        except BaseException as exc:
            if not isinstance(exc, tuple(policy.retry_on)):
                raise
            last_exc = exc
            if attempt >= policy.attempts:
                break
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


class ResilientHTTPClient:
    """httpx client that applies a per-host breaker plus retry/backoff."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        registry: Optional[BreakerRegistry] = None,
        policy: Optional[RetryPolicy] = None,
        retry_statuses: Iterable[int] = (502, 503, 504),
    ):
        self._client = client
        self._registry = registry or BreakerRegistry()
        base_policy = policy or RetryPolicy()
        self._policy = RetryPolicy(
            attempts=base_policy.attempts,
            base_delay=base_policy.base_delay,
            max_delay=base_policy.max_delay,
            jitter=base_policy.jitter,
            retry_on=tuple(base_policy.retry_on) + (RetryableStatusError,),
        )
        self._retry_statuses = set(retry_statuses)

    @property
    def registry(self) -> BreakerRegistry:
        return self._registry

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        host = httpx.URL(url).host or url
        breaker = self._registry.get(host)

        async def attempt() -> httpx.Response:
            response = await self._client.request(method, url, **kwargs)
            if response.status_code in self._retry_statuses:
                raise RetryableStatusError(
                    f"upstream returned {response.status_code}",
                    request=response.request,
                    response=response,
                )
            return response

        async def guarded() -> httpx.Response:
            return await retry_async(attempt, self._policy)

        return await breaker.call(guarded)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()
