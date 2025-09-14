from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class DeliveryRequest(BaseModel):
    session_id: str
    request_id: str
    destination: str
    delivery_method: str  # "API", "MQTT", "HSM", "FILE"
    key_material: str
    algorithm: str
    # Negotiator envia epoch (int). Mantemos int para evitar erro de validação.
    expires_at: int = Field(..., description="Unix epoch (seconds)")
    metadata: Optional[Dict[str, Any]] = None

class DeliveryResponse(BaseModel):
    session_id: str
    request_id: str
    destination: str
    status: str                 # "delivered" | "failed" | "pending"
    delivery_method: str
    timestamp: datetime
    delivery_id: str            # ID único da entrega
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