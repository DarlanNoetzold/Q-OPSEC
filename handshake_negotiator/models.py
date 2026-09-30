from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class NegotiationRequest(BaseModel):
    request_id: Optional[str] = None
    source: str
    destination: str
    proposed: List[str]
    dst_props: Optional[Dict[str, Any]] = None

    @field_validator("source", "destination", mode="before")
    @classmethod
    def reject_blank_endpoints(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("source and destination must not be blank")
        return value

    @field_validator("proposed")
    @classmethod
    def validate_proposed_algorithms(cls, value):
        if not value:
            raise ValueError("proposed must contain at least one algorithm")
        if any(not isinstance(algorithm, str) or not algorithm.strip() for algorithm in value):
            raise ValueError("proposed algorithms must not be blank")
        return value

    class Config:
        populate_by_name = True
        extra = "allow"

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

    crypto_nonce_b64: Optional[str] = None
    crypto_ciphertext_b64: Optional[str] = None
    crypto_algorithm: Optional[str] = None
    crypto_expires_at: Optional[int] = None