from pydantic import BaseModel, Field
from typing import Optional

class EncryptRequest(BaseModel):
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    algorithm: str = Field(..., description="AES256_GCM | CHACHA20_POLY1305")

    plaintext_b64: Optional[str] = None

    aad_b64: Optional[str] = None

    fetch_from_interceptor: bool = True

class EncryptResponse(BaseModel):
    session_id: str
    algorithm: str
    nonce_b64: str
    ciphertext_b64: str
    expires_at: int

class DecryptRequest(BaseModel):
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    algorithm: str = Field(..., description="AES256_GCM | CHACHA20_POLY1305")
    nonce_b64: str
    ciphertext_b64: str
    aad_b64: Optional[str] = None

class DecryptResponse(BaseModel):
    session_id: str
    algorithm: str
    plaintext_b64: str

class ErrorResponse(BaseModel):
    detail: str