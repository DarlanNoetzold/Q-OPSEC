from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class NegotiationRequest(BaseModel):
    request_id: str
    source: str
    destination: str
    proposed: List[str]
    dst_props: Optional[Dict[str, Any]] = None

class NegotiationResponse(BaseModel):
    request_id: str
    session_id: str
    requested_algorithm: str
    selected_algorithm: str
    key_material: str
    expires_at: datetime
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    source_of_key: str
    message: Optional[str] = None
    delivery_id: Optional[str] = None
    delivery_status: Optional[str] = None