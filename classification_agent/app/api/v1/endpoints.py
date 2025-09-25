"""
API endpoints for the Classification Agent.
"""
import time
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ...core.config import settings
from ...core.security import get_current_user, require_auth
from ...core.logging import get_logger
from ...models.schemas import (
    HealthResponse, ModelInfo, PredictRequest, PredictResponse,
    ModelReloadRequest, ModelReloadResponse, MetricsResponse,
    PredictionResult
)
from ...services.model_service import model_service, ModelLoadError, PredictionError
from ...services.metrics_service import metrics_service
from ...utils.exceptions import (
    ModelNotLoadedException, PredictionException, ValidationException
)

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    health_data = metrics_service.get_health_status()

    return HealthResponse(
        status=health_data["status"],
        version=settings.api_version,
        model_loaded=health_data["model_loaded"],
        model_name=health_data.get("model_name"),
        uptime_seconds=health_data["uptime_seconds"]
    )


@router.get("/model", response_model=ModelInfo, tags=["Model"])
async def get_model_info(user: Dict[str, Any] = Depends(get_current_user)):
    """Get information about the currently loaded model."""
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    model_info = model_service.get_model_info()
    if not model_info:
        raise ModelNotLoadedException("Model information not available")

    return model_info


@router.post("/model/reload", response_model=ModelReloadResponse, tags=["Model"])
async def reload_model(
        request: ModelReloadRequest = ModelReloadRequest(),
        user: Dict[str, Any] = Depends(require_auth)
):
    """Reload the model from registry."""
    try:
        previous_model = model_service.model_name if model_service.is_model_loaded() else None

        # Attempt to reload
        reloaded = model_service.load_latest_model(force=request.force)

        if reloaded:
            metrics_service.record_model_reload(model_service.model_name)
            message = "Model reloaded successfully"
            logger.info("Model reloaded",
                        new_model=model_service.model_name,
                        previous_model=previous_model)
        else:
            message = "Model was already up to date"

        return ModelReloadResponse(
            status="success",
            model_name=model_service.model_name,
            previous_model=previous_model,
            message=message
        )

    except ModelLoadError as e:
        logger.error("Model reload failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
        request: PredictRequest,
        http_request: Request,
        user: Dict[str, Any] = Depends(get_current_user)
):
    """Make predictions on input data."""
    start_time = time.time()

    # Check if model is loaded
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    try:
        # Validate input data
        validation_errors = model_service.validate_input(request.data)
        if validation_errors:
            raise ValidationException(
                "Input validation failed",
                details={"validation_errors": validation_errors}
            )

        # Make predictions
        labels, probabilities, input_hashes = model_service.predict(
            request.data,
            include_probabilities=request.include_probabilities
        )

        # Calculate batch size
        batch_size = 1 if isinstance(request.data, dict) else len(request.data)

        # Record metrics
        metrics_service.record_prediction(model_service.model_name, batch_size)

        # Build response
        results = []
        for i, label in enumerate(labels):
            confidence = None
            probs = None

            if probabilities and i < len(probabilities):
                probs = probabilities[i]
                # Confidence is the probability of the predicted class
                confidence = probs.get(label, 0.0)

            result = PredictionResult(
                label=label,
                confidence=confidence,
                probabilities=probs,
                input_hash=input_hashes[i] if i < len(input_hashes) else None
            )
            results.append(result)

        prediction_time = (time.time() - start_time) * 1000  # Convert to ms

        response = PredictResponse(
            results=results,
            model_name=model_service.model_name,
            model_version=None,  # Could be added to model metadata
            classes=model_service.classes,
            prediction_time_ms=round(prediction_time, 2),
            batch_size=batch_size
        )

        logger.info(
            "Prediction completed",
            request_id=getattr(http_request.state, 'request_id', None),
            model_name=model_service.model_name,
            batch_size=batch_size,
            prediction_time_ms=prediction_time
        )

        return response

    except PredictionError as e:
        logger.error("Prediction failed", error=str(e))
        raise PredictionException(str(e))
    except Exception as e:
        logger.error("Unexpected error during prediction", error=str(e), exc_info=True)
        raise PredictionException(f"Prediction failed: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(user: Dict[str, Any] = Depends(require_auth)):
    """Get API metrics and statistics."""
    metrics_data = metrics_service.get_metrics()

    return MetricsResponse(
        total_requests=metrics_data["total_requests"],
        total_predictions=metrics_data["total_predictions"],
        average_response_time_ms=metrics_data["average_response_time_ms"],
        error_rate=metrics_data["error_rate"],
        model_reload_count=metrics_data["model_reload_count"],
        uptime_seconds=metrics_data["uptime_seconds"],
        last_prediction_at=metrics_data["last_prediction_at"],
        current_model=metrics_data["current_model"]
    )


@router.get("/model/manifest", tags=["Model"])
async def get_model_manifest(user: Dict[str, Any] = Depends(get_current_user)):
    """Get detailed model manifest including required columns and types."""
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    model_info = model_service.get_model_info()
    if not model_info:
        raise ModelNotLoadedException("Model information not available")

    # Additional manifest information
    manifest = {
        "model_name": model_info.saved_model_name,
        "classes": model_info.classes,
        "required_columns": model_info.required_columns,
        "input_schema": {
            "type": "object",
            "properties": {col: {"type": "any"} for col in model_info.required_columns},
            "required": model_info.required_columns
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": model_info.classes},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "probabilities": {
                    "type": "object",
                    "properties": {cls: {"type": "number"} for cls in model_info.classes}
                }
            }
        },
        "metadata": model_info.meta,
        "loaded_at": model_info.loaded_at
    }

    return manifest


@router.post("/validate", tags=["Validation"])
async def validate_input(
        request: PredictRequest,
        user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate input data without making predictions."""
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    validation_errors = model_service.validate_input(request.data)

    batch_size = 1 if isinstance(request.data, dict) else len(request.data)

    return {
        "valid": len(validation_errors) == 0,
        "errors": validation_errors,
        "batch_size": batch_size,
        "required_columns": model_service.required_columns,
        "model_name": model_service.model_name
    }