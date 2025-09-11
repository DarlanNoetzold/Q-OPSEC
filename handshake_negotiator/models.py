from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from typing import List, Optional, Dict, Any

class NegotiationRequest(BaseModel):
    source: str
    destination: str
    proposed: List[str]
    dst_props: Optional[Dict[str, Any]] = None

class NegotiationResponse(BaseModel):
    negotiated: str
    session_id: str
    key_material: str
    expires_at: datetime
    fallback_used: bool = False
    message: Optional[str] = None