from pydantic import BaseModel
from typing import Optional, Dict, Any


class CreateKeyRequest(BaseModel):
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    algorithm: str
    ttl_seconds: int = 3600
    strict: bool = False


class CreateKeyResponse(BaseModel):
    session_id: str
    request_id: str
    requested_algorithm: str
    selected_algorithm: str
    key_material: str
    expires_at: int
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    source_of_key: str
    qkd_metadata: Optional[Dict[str, Any]] = None


class KeyResponse(BaseModel):
    session_id: str
    request_id: Optional[str] = None
    algorithm: str
    key_material: str
    expires_at: int
