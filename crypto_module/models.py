from pydantic import BaseModel, Field, field_validator
from typing import Optional


class EncryptRequest(BaseModel):
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    algorithm: str = Field(..., description="AEAD name or negotiated KEM/signature name")

    plaintext_b64: Optional[str] = None

    aad_b64: Optional[str] = None

    fetch_from_interceptor: bool = True

    @field_validator("algorithm", "session_id", "request_id", mode="before")
    @classmethod
    def reject_blank_text(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("text fields must not be blank")
        if isinstance(value, str) and len(value) > 256:
            raise ValueError("text fields must not exceed 256 characters")
        return value

class EncryptResponse(BaseModel):
    session_id: str
    algorithm: str
    nonce_b64: str
    ciphertext_b64: str
    expires_at: int

class DecryptRequest(BaseModel):
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    algorithm: str = Field(..., description="AEAD name or negotiated KEM/signature name")
    nonce_b64: str
    ciphertext_b64: str
    aad_b64: Optional[str] = None

    @field_validator("algorithm", "session_id", "request_id", mode="before")
    @classmethod
    def reject_blank_text(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("text fields must not be blank")
        if isinstance(value, str) and len(value) > 256:
            raise ValueError("text fields must not exceed 256 characters")
        return value

class DecryptResponse(BaseModel):
    session_id: str
    algorithm: str
    plaintext_b64: str

class ErrorResponse(BaseModel):
    detail: str