"""
Custom middleware for the API.
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.logging import get_logger
from ..services.metrics_service import metrics_service

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time = time.time() - start_time

        # Record metrics
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code

        metrics_service.record_request(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time
        )

        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time * 1000:.2f}ms"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log the error
            request_id = getattr(request.state, 'request_id', 'unknown')

            # Evita logging de HTTPException (já são tratadas)
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                # Re-raise HTTPException para ser tratada pelo FastAPI
                raise

            logger.error(
                "Unhandled exception",
                request_id=request_id,
                path=request.url.path,
                method=request.method,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Record error metric
            metrics_service.record_error("unhandled_exception")

            # Return JSON response instead of raising
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                    "error_type": type(e).__name__
                },
                headers={"X-Request-ID": request_id}
            )


class CORSMiddleware:
    """Custom CORS middleware configuration."""

    @staticmethod
    def get_cors_config():
        """Get CORS configuration."""
        from fastapi.middleware.cors import CORSMiddleware
        from ..core.config import settings

        return {
            "middleware_class": CORSMiddleware,
            "allow_origins": settings.allowed_hosts,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["*"],
            "expose_headers": ["X-Request-ID", "X-Response-Time"]
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # client_ip -> list of request timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old requests (older than 1 minute)
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.requests[client_ip] = []

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )

        # Add current request
        self.requests[client_ip].append(current_time)

        return await call_next(request)