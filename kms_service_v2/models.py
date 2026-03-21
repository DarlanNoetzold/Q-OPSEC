"""
Pydantic models for KMS API requests and responses.
"""
from pydantic import BaseModel, model_validator
from typing import Optional


class CreateKeyRequest(BaseModel):
    """Request body for POST /kms/create_key"""
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    algorithm: str
    ttl_seconds: int = 3600
    strict: bool = False

    @model_validator(mode="after")
    def validate_ids(self):
        if not self.session_id and not self.request_id:
            raise ValueError("Either session_id or request_id must be provided")
        return self


class CreateKeyResponse(BaseModel):
    """Response body for POST /kms/create_key"""
    session_id: str
    request_id: str
    requested_algorithm: str
    selected_algorithm: str
    key_material: str
    expires_at: int
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    source_of_key: str  # "qkd" | "pqc" | "classical"


class KeyResponse(BaseModel):
    """Response body for GET /kms/get_key/{id}"""
    session_id: str
    request_id: str
    algorithm: str
    key_material: str
    expires_at: int
