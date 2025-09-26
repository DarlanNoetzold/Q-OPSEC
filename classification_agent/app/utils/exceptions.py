"""
Custom exceptions and error handlers.
"""
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from ..core.logging import get_logger
from ..models.database import ErrorResponse, ValidationErrorResponse, ValidationErrorDetail
from ..services.metrics_service import metrics_service

logger = get_logger(__name__)


class ClassificationAgentException(Exception):
    """Base exception for Classification Agent."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedException(ClassificationAgentException):
    """Exception raised when no model is loaded."""
    pass


class ModelLoadException(ClassificationAgentException):
    """Exception raised when model loading fails."""
    pass


class PredictionException(ClassificationAgentException):
    """Exception raised when prediction fails."""
    pass


class ValidationException(ClassificationAgentException):
    """Exception raised for validation errors."""
    pass


async def classification_agent_exception_handler(
    request: Request,
    exc: ClassificationAgentException
) -> JSONResponse:
    """Handle custom Classification Agent exceptions."""
    request_id = getattr(request.state, 'request_id', None)

    # Log the error
    logger.error(
        "Classification Agent exception",
        request_id=request_id,
        exception_type=type(exc).__name__,
        message=exc.message,
        details=exc.details
    )

    # Record error metric
    metrics_service.record_error(type(exc).__name__)

    # Determine status code based on exception type
    if isinstance(exc, ModelNotLoadedException):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, ValidationException):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, PredictionException):
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    error_response = ErrorResponse(
        error=type(exc).__name__,
        message=exc.message,
        details=exc.details,
        request_id=request_id
    )

    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(error_response.dict())
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, 'request_id', None)

    logger.warning(
        "HTTP exception",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail
    )

    metrics_service.record_error("http_exception")

    error_response = ErrorResponse(
        error="http_error",
        message=str(exc.detail),
        request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_response.dict())
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    request_id = getattr(request.state, 'request_id', None)

    logger.warning(
        "Validation error",
        request_id=request_id,
        errors=exc.errors()
    )

    metrics_service.record_error("validation_error")

    # Convert validation errors to our format
    validation_details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        validation_details.append(
            ValidationErrorDetail(
                field=field,
                message=error["msg"],
                value=error.get("input")
            )
        )

    error_response = ValidationErrorResponse(
        details=validation_details
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(error_response.dict())
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    request_id = getattr(request.state, 'request_id', None)

    logger.error(
        "Unhandled exception",
        request_id=request_id,
        exception_type=type(exc).__name__,
        message=str(exc),
        exc_info=True
    )

    metrics_service.record_error("unhandled_exception")

    error_response = ErrorResponse(
        error="internal_error",
        message="An internal error occurred",
        request_id=request_id
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(error_response.dict())
    )


def setup_exception_handlers(app):
    """Setup all exception handlers for the FastAPI app."""
    app.add_exception_handler(ClassificationAgentException, classification_agent_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)