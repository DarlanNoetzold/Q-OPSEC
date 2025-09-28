"""
API endpoints for the Classification Agent.
"""
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
import requests

from ...core.config import settings
from ...core.security import get_current_user, require_auth
from ...core.logging import get_logger
from ...services.model_service import model_service, ModelLoadError, PredictionError
from ...services.metrics_service import metrics_service
from ...utils.exceptions import (
    ModelNotLoadedException, PredictionException, ValidationException
)

logger = get_logger(__name__)
router = APIRouter()


# Schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    uptime_seconds: float
    model_config = {"protected_namespaces": ()}


class ModelReloadRequest(BaseModel):
    force: bool = False
    model_config = {"protected_namespaces": ()}


class ModelReloadResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    previous_model: Optional[str] = None
    message: str
    model_config = {"protected_namespaces": ()}


class PredictionRequest(BaseModel):
    data: Any = Field(..., description="Objeto ou lista de objetos com os dados para predição")
    return_probabilities: bool = True
    send_to_rl: bool = False
    model_config = {"protected_namespaces": ()}


class PredictionResult(BaseModel):
    label: str
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    input_hash: Optional[str] = None
    rl_decision: Optional[str] = None   # 🔥 Novo campo
    model_config = {"protected_namespaces": ()}


class PredictResponse(BaseModel):
    results: List[PredictionResult]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    classes: List[str] = []
    prediction_time_ms: float
    batch_size: int
    model_config = {"protected_namespaces": ()}


class MetricsResponse(BaseModel):
    total_requests: int
    total_predictions: int
    average_response_time_ms: float
    error_rate: float
    model_reload_count: int
    uptime_seconds: float
    last_prediction_at: Optional[datetime] = None
    current_model: Optional[str] = None
    model_config = {"protected_namespaces": ()}


# -------------------- Endpoints --------------------

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    health_data = metrics_service.get_health_status()
    return HealthResponse(
        status=health_data.get("status", "unknown"),
        version=settings.api_version,
        model_loaded=health_data.get("model_loaded", model_service.is_model_loaded()),
        model_name=health_data.get("model_name", model_service.model_name),
        uptime_seconds=health_data.get("uptime_seconds", 0.0),
    )


@router.get("/model", tags=["Model"])
async def get_model_info(user: Dict[str, Any] = Depends(get_current_user)):
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    info = model_service.get_model_info()
    if not info:
        raise ModelNotLoadedException("Model information not available")

    return info


@router.post("/model/reload", response_model=ModelReloadResponse, tags=["Model"])
async def reload_model(
    request: ModelReloadRequest = ModelReloadRequest(),
    user: Dict[str, Any] = Depends(require_auth),
):
    try:
        previous_model = model_service.model_name if model_service.is_model_loaded() else None
        reloaded = await model_service.load_latest_model(force=request.force)

        if reloaded:
            metrics_service.record_model_reload(model_service.model_name)
            message = "Model reloaded successfully"
            logger.info("Model reloaded", new_model=model_service.model_name, previous_model=previous_model)
        else:
            message = "Model was already up to date"

        return ModelReloadResponse(
            status="success",
            model_name=model_service.model_name,
            previous_model=previous_model,
            message=message,
        )

    except ModelLoadError as e:
        logger.error("Model reload failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        )


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    start_time = time.time()

    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    try:
        validation_errors = model_service.validate_input(request.data)
        if validation_errors:
            raise ValidationException(
                "Input validation failed",
                details={"validation_errors": validation_errors},
            )

        labels, probabilities, input_hashes = model_service.predict(
            request.data, include_probabilities=request.return_probabilities
        )

        batch_size = 1 if isinstance(request.data, dict) else len(request.data)
        metrics_service.record_prediction(model_service.model_name, batch_size)

        rl_decisions: List[Optional[str]] = [None] * batch_size

        if request.send_to_rl:
            RL_ENGINE_URL = getattr(settings, "rl_engine_url", "http://localhost:9009/act")

            rl_payload = {
                "request_id": request.data.get("request_id_resolved"),
                "source": "node-A",  # pode vir do context_api ou settings
                "destination": "http://localhost:9000",
                "security_level": labels[0],  # pegar o resultado da predição
                "risk_score": request.data.get("risk_score", 0.0),
                "conf_score": request.data.get("conf_score", 0.0),
                "dst_props": {
                    "hardware": ["QKD"],  # ou algo derivado de request.data
                    "compliance": ["GDPR"],
                    "max_latency_ms": 10
                }
            }

            try:
                rl_response = requests.post(RL_ENGINE_URL, json=rl_payload)
                rl_response.raise_for_status()
                rl_result = rl_response.json()
                print(rl_result)
            except Exception as e:
                logger.warning("RL Engine unavailable, fallback ignored", error=str(e))
                rl_result = {"error": str(e)}

        results: List[PredictionResult] = []
        for i, label in enumerate(labels):
            probs = probabilities[i] if probabilities and i < len(probabilities) else None
            conf = probs.get(label) if probs else None
            results.append(
                PredictionResult(
                    label=str(label),
                    confidence=conf,
                    probabilities=probs,
                    input_hash=input_hashes[i] if i < len(input_hashes) else None,
                    rl_decision=rl_decisions[i],
                )
            )

        prediction_time = (time.time() - start_time) * 1000.0

        resp = PredictResponse(
            results=results,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            classes=model_service.classes or [],
            prediction_time_ms=round(prediction_time, 2),
            batch_size=batch_size,
        )

        logger.info(
            "Prediction completed",
            request_id=getattr(http_request.state, "request_id", None),
            model_name=model_service.model_name,
            batch_size=batch_size,
            prediction_time_ms=resp.prediction_time_ms,
        )
        return resp

    except PredictionError as e:
        logger.error("Prediction failed", error=str(e))
        raise PredictionException(str(e))
    except Exception as e:
        logger.error("Unexpected error during prediction", error=str(e), exc_info=True)
        raise PredictionException(f"Prediction failed: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(user: Dict[str, Any] = Depends(require_auth)):
    m = metrics_service.get_metrics()
    return MetricsResponse(
        total_requests=m.get("total_requests", 0),
        total_predictions=m.get("total_predictions", 0),
        average_response_time_ms=m.get("average_response_time_ms", 0.0),
        error_rate=m.get("error_rate", 0.0),
        model_reload_count=m.get("model_reload_count", 0),
        uptime_seconds=m.get("uptime_seconds", 0.0),
        last_prediction_at=m.get("last_prediction_at"),
        current_model=m.get("current_model"),
    )


@router.get("/model/manifest", tags=["Model"])
async def get_model_manifest(user: Dict[str, Any] = Depends(get_current_user)):
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    info = model_service.get_model_info()
    if not info:
        raise ModelNotLoadedException("Model information not available")

    classes = info.get("classes", [])
    required_columns = info.get("required_columns", [])
    manifest = {
        "model_name": info.get("saved_model_name") or info.get("model_name") or model_service.model_name,
        "classes": classes,
        "required_columns": required_columns,
        "input_schema": {
            "type": "object",
            "properties": {col: {"type": "any"} for col in required_columns},
            "required": required_columns,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": classes},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "probabilities": {
                    "type": "object",
                    "properties": {cls: {"type": "number"} for cls in classes},
                },
                "rl_decision": {"type": ["string", "null"]},
            },
        },
        "metadata": info.get("meta", {}),
        "loaded_at": info.get("loaded_at"),
    }
    return manifest