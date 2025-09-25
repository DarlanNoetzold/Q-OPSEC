"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.api.v1 import router as v1_router
from app.utils.middleware import (
    RequestIDMiddleware,
    MetricsMiddleware,
    ErrorHandlingMiddleware,
    CORSMiddleware as CustomCORSMiddleware,
    RateLimitMiddleware
)
from app.utils.exceptions import setup_exception_handlers
from app.services.model_service import model_service
from app.services.metrics_service import metrics_service

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Classification Agent API", version=settings.api_version)

    # Try to load model on startup
    try:
        model_service.load_latest_model()
        logger.info("Model loaded successfully on startup", model_name=model_service.model_name)
    except Exception as e:
        logger.warning("Failed to load model on startup", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down Classification Agent API")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Setup exception handlers
setup_exception_handlers(app)

# Add middleware (order matters - first added is outermost)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)

# Add CORS middleware
cors_config = CustomCORSMiddleware.get_cors_config()
app.add_middleware(
    cors_config["middleware_class"],
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
    expose_headers=cors_config["expose_headers"]
)

# Add rate limiting if enabled
if settings.rate_limit_requests > 0:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests
    )

# Include API routes
app.include_router(
    v1_router,
    prefix=settings.api_prefix,
    tags=["v1"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs_url": "/docs",
        "health_url": f"{settings.api_prefix}/health",
        "model_loaded": model_service.is_model_loaded(),
        "model_name": model_service.model_name if model_service.is_model_loaded() else None
    }


# Additional health endpoint at root level
@app.get("/health", tags=["Health"])
async def root_health():
    """Simple health check at root level."""
    return {
        "status": "ok" if model_service.is_model_loaded() else "degraded",
        "model_loaded": model_service.is_model_loaded()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )