"""
Structured logging configuration.
"""
import sys
import structlog
from typing import Any, Dict
from pathlib import Path

from .config import settings


def configure_logging() -> None:
    """Configure structured logging."""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.is_development else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, settings.log_level.upper(), structlog.stdlib.INFO)
        ),
        logger_factory=structlog.WriteLoggerFactory(
            file=open(settings.log_file, "a") if settings.log_file else sys.stdout
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests."""

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = scope.get("request_id", "unknown")
        method = scope["method"]
        path = scope["path"]

        # Log request start
        self.logger.info(
            "Request started",
            request_id=request_id,
            method=method,
            path=path,
            client=scope.get("client", ["unknown", 0])[0]
        )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                self.logger.info(
                    "Request completed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)