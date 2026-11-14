from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

DEFAULT_HOST_REWRITES = "10.0.0.5=192.168.18.18"


class SharedAsyncClient:
    """Lazily created httpx.AsyncClient reused across requests (one connection pool per service)."""

    def __init__(self, **client_kwargs: Any):
        self._client_kwargs = client_kwargs
        self._client: httpx.AsyncClient | None = None

    def get(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(**self._client_kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


def parse_host_rewrites(raw: str | None) -> dict[str, str]:
    rewrites: dict[str, str] = {}
    for pair in (raw or "").split(","):
        source, sep, target = pair.strip().partition("=")
        if sep and source.strip() and target.strip():
            rewrites[source.strip()] = target.strip()
    return rewrites


def host_rewrites() -> dict[str, str]:
    return parse_host_rewrites(os.getenv("QOPSEC_HOST_REWRITES", DEFAULT_HOST_REWRITES))


def rewrite_host(url: str, rewrites: Mapping[str, str] | None = None) -> str:
    """Map lab-internal hosts to reachable ones (configured via QOPSEC_HOST_REWRITES='a=b,c=d')."""
    for source, target in (host_rewrites() if rewrites is None else rewrites).items():
        url = url.replace(source, target)
    return url
