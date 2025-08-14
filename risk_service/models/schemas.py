# models/schemas.py
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
from datetime import datetime

class GeneralSignals(BaseModel):
    global_alert_level: str = Field(..., description="none, low, medium, high, critical")
    current_campaigns: List[Dict[str, str]] = Field(default_factory=list, description="[{name, severity, geo, target_type}]")
    anomaly_index_global: float = Field(0.0, ge=0.0, le=1.0)
    incident_rate_7d: int = Field(0)
    patch_delay_days_p50: int = Field(0)
    exposure_level: str = Field("medium", description="low, medium, high")
    maintenance_window: bool = Field(False)
    compliance_debt_score: float = Field(0.0, ge=0.0, le=1.0)
    business_critical_period: bool = Field(False)
    geo_region: Optional[str] = Field(None)

class AssessRequest(BaseModel):
    request_id: str
    signals: GeneralSignals

class RiskContext(BaseModel):
    score: float
    level: str  # very_low, low, medium, high, critical
    anomaly_score: float
    threat_intel: Dict[str, object]
    recent_incidents: int
    policy_overrides: List[str]
    timestamp: datetime
    model_version: str

class TrainRequest(BaseModel):
    n: int = Field(400, ge=50, le=5000)
    seed: Optional[int] = 42

class TrainResponse(BaseModel):
    model_version: str
    metrics: Dict[str, float]
    samples: int

def validate_payload(model_cls, payload):
    try:
        return model_cls(**payload), None
    except ValidationError as e:
        return None, e