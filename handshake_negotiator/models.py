from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class NegotiationRequest(BaseModel):
    source: str
    destination: str
    proposed: List[str]
    dst_props: Optional[Dict[str, Any]] = None

class NegotiationResponse(BaseModel):
    session_id: str
    requested_algorithm: str      # O que foi pedido originalmente
    selected_algorithm: str       # O que foi realmente usado
    key_material: str
    expires_at: datetime
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    source_of_key: str           # "qkd" | "pqc" | "classical"
    message: Optional[str] = None