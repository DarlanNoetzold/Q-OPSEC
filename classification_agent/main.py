# main.py
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.core.config import settings
from app.services.database import db_service
from app.services.model_service import model_service
from app.api.v1.endpoints import router
from app.utils.middleware import (
    RequestIDMiddleware,
    MetricsMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware
)
from app.utils.exceptions import setup_exception_handlers
from app.core.logging import configure_logging

# Configurar logging
configure_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    # Startup
    logger.info("Starting Classification Agent API")

    try:
        # Conecta ao MongoDB
        await db_service.connect()
        logger.info("Database connected successfully")

        # Carrega o modelo mais recente
        try:
            await model_service.load_latest_model()  # Agora é async
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load model on startup", error=str(e))

        yield

    finally:
        # Shutdown
        logger.info("Shutting down Classification Agent API")
        await db_service.disconnect()


def create_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI"""

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middlewares
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Exception handlers
    setup_exception_handlers(app)

    # Rotas
    app.include_router(router, prefix=settings.api_prefix)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )