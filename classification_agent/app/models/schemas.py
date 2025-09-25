"""
Pydantic models for API request/response schemas.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


class ModelInfo(BaseModel):
    """Information about the loaded model."""
    saved_model_name: str = Field(..., description="Name of the saved model")
    artifact_path: str = Field(..., description="Path to model artifact")
    classes: List[str] = Field(..., description="Model output classes")
    required_columns: List[str] = Field(..., description="Required input columns")
    cv_metrics: Optional[ModelMetrics] = Field(None, description="Cross-validation metrics")
    holdout_metrics: Optional[ModelMetrics] = Field(None, description="Holdout test metrics")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Model metadata")
    loaded_at: datetime = Field(default_factory=datetime.utcnow)
    registry_dir: str = Field(..., description="Model registry directory")
    model_dir: str = Field(..., description="Specific model directory")


class PredictionInput(BaseModel):
    """Single prediction input record."""
    # Dynamic fields - will be validated against model requirements
    data: Dict[str, Any] = Field(..., description="Input features for prediction")

    @validator('data')
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError("Input data cannot be empty")
        return v


class PredictRequest(BaseModel):
    """Prediction request payload."""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="Single record (dict) or batch of records (list of dicts)"
    )
    include_probabilities: bool = Field(
        True,
        description="Whether to include class probabilities in response"
    )

    @validator('data')
    def validate_data_structure(cls, v):
        if isinstance(v, dict):
            if not v:
                raise ValueError("Input data cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input data list cannot be empty")
            if len(v) > 1000:  # Configurable limit
                raise ValueError("Batch size too large (max 1000 records)")
            for i, record in enumerate(v):
                if not isinstance(record, dict) or not record:
                    raise ValueError(f"Record {i} must be a non-empty dictionary")
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries")
        return v


class PredictionResult(BaseModel):
    """Single prediction result."""
    label: str = Field(..., description="Predicted class label")
    confidence: Optional[float] = Field(None, description="Confidence score for predicted class")
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Class probabilities (if available)"
    )
    input_hash: Optional[str] = Field(None, description="Hash of input data for tracking")


class PredictResponse(BaseModel):
    """Prediction response."""
    results: List[PredictionResult] = Field(..., description="Prediction results")
    model_name: str = Field(..., description="Name of model used for prediction")
    model_version: Optional[str] = Field(None, description="Version of model used")
    classes: List[str] = Field(..., description="All possible output classes")
    prediction_time_ms: Optional[float] = Field(None, description="Prediction time in milliseconds")
    batch_size: int = Field(..., description="Number of records processed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelReloadRequest(BaseModel):
    """Model reload request."""
    force: bool = Field(False, description="Force reload even if model hasn't changed")


class ModelReloadResponse(BaseModel):
    """Model reload response."""
    status: str = Field(..., description="Reload status")
    model_name: str = Field(..., description="Name of loaded model")
    previous_model: Optional[str] = Field(None, description="Previously loaded model")
    reloaded_at: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = Field(None, description="Additional information")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Error message")
    value: Any = Field(None, description="Invalid value")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "validation_error"
    message: str = "Request validation failed"
    details: List[ValidationErrorDetail] = Field(..., description="Validation error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """API metrics response."""
    total_requests: int = Field(..., description="Total number of requests")
    total_predictions: int = Field(..., description="Total number of predictions made")
    average_response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate (0-1)")
    model_reload_count: int = Field(..., description="Number of model reloads")
    uptime_seconds: float = Field(..., description="Service uptime")
    last_prediction_at: Optional[datetime] = Field(None, description="Timestamp of last prediction")
    current_model: Optional[str] = Field(None, description="Currently loaded model")