from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
from datetime import datetime

# Input model from Context API -> this service
class SourceContext(BaseModel):
    ip: Optional[str] = None

class DestinationContext(BaseModel):
    service_id: Optional[str] = None

class ContentPointer(BaseModel):
    # Opaque reference for production. For this PoC we allow an optional sample_text.
    ref: Optional[str] = Field(default=None, description="Opaque reference or ID")
    sample_text: Optional[str] = Field(default=None, description="Small safe sample to classify")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Optional metadata (doc_type, app, etc.)")

class ClassifyRequest(BaseModel):
    request_id: str
    content_pointer: ContentPointer
    source: Optional[SourceContext] = None
    destination: Optional[DestinationContext] = None

class ContentConfidentiality(BaseModel):
    classification: str  # public, internal, confidential, restricted
    score: float
    tags: List[str]
    detected_patterns: List[str]
    dlp_findings: List[Dict[str, object]]
    source_app_context: Optional[str] = None
    user_label: Optional[str] = None
    model_version: str

class TrainRequest(BaseModel):
    n_per_class: int = Field(80, ge=20, le=2000)
    seed: Optional[int] = 42
    vocab_noise: int = Field(100, ge=0, le=500, description="Extra random tokens to improve robustness")

class TrainResponse(BaseModel):
    model_version: str
    metrics: Dict[str, float]
    samples: int

def validate_payload(model_cls, payload):
    try:
        return model_cls(**payload), None
    except ValidationError as e:
        return None, e