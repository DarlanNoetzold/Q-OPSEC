from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from datetime import datetime

class DeliveryRequest(BaseModel):
    session_id: str
    request_id: Optional[str] = None
    destination: str
    delivery_method: str
    key_material: str
    algorithm: str
    expires_at: int = Field(..., description="Unix epoch (seconds)")
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("session_id", "request_id", "destination", "delivery_method", "key_material", "algorithm", mode="before")
    @classmethod
    def reject_blank_text(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("required text fields must not be blank")
        return value

    class Config:
        populate_by_name = True
        extra = "allow"

class DeliveryResponse(BaseModel):
    session_id: str
    request_id: str
    destination: str
    status: str
    delivery_method: str
    timestamp: datetime
    delivery_id: str
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DeliveryStatus(BaseModel):
    delivery_id: str
    session_id: str
    request_id: str
    status: str
    last_attempt: datetime
    attempts: int
    next_retry: Optional[datetime] = None