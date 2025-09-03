from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import uuid


class HandshakeRequest(BaseModel):
    source: str = Field(..., example="node-A")
    destination: str = Field(..., example="node-B")
    proposed: List[str] = Field(..., example=["QKD_BB84", "Kyber1024", "AES256_GCM"])


class HandshakeResponse(BaseModel):
    negotiated: str
    session_key_id: str
    expires_at: datetime
    fallback_used: Optional[bool] = False
    message: str