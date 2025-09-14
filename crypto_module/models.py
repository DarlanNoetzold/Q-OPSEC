from pydantic import BaseModel, Field
from typing import Optional

class EncryptRequest(BaseModel):
    # Identificação do contexto (pelo menos um deve ser fornecido)
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    # Algoritmo de cifra pretendido
    algorithm: str = Field(..., description="AES256_GCM | CHACHA20_POLY1305")

    # Dados do payload em base64 (opcional se for buscar no Context API via request_id)
    plaintext_b64: Optional[str] = None

    # Associated Authenticated Data opcional (base64). Se ausente, um AAD padrão será usado.
    aad_b64: Optional[str] = None

    # Quando True (default), se plaintext_b64 não for enviado, busca no Context API via request_id
    fetch_from_context: bool = True

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