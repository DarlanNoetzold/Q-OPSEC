from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class NegotiationRequest(BaseModel):
    source: str
    destination: str
    proposed: List[str]

class NegotiationResponse(BaseModel):
    negotiated: str
    session_id: str
    key_material: str
    expires_at: datetime
    fallback_used: bool = False
    message: Optional[str] = None