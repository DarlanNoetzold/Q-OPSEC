from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any


class CreateKeyRequest(BaseModel):
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    algorithm: str
    ttl_seconds: int = 3600
    strict: bool = False

    @field_validator("algorithm", "session_id", "request_id", mode="before")
    @classmethod
    def normalize_text_fields(cls, value):
        if value is None:
            return value
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must not be blank")
        if len(normalized) > 256:
            raise ValueError("text fields must be 256 characters or fewer")
        return normalized


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
