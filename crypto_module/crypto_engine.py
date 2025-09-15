import base64
import os
import httpx
from datetime import datetime
from typing import Optional, Tuple

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from config import (
    KMS_BASE_URL,
    HTTP_TIMEOUT,
    AESGCM_NONCE_SIZE,
    CHACHA20_NONCE_SIZE,
    HKDF_SALT,
    HKDF_INFO_ENCRYPT,
    HKDF_INFO_DECRYPT,
)

# ---- Helpers ----

def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def b64d(data_b64: Optional[str]) -> Optional[bytes]:
    if data_b64 is None:
        return None
    return base64.b64decode(data_b64.encode("utf-8"))

def _hkdf_derive(key_material_b64: str, length: int, info: bytes) -> bytes:
    key_bytes = b64d(key_material_b64)
    if not key_bytes:
        raise ValueError("Empty key_material from KMS")

    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=HKDF_SALT, info=info)
    return hkdf.derive(key_bytes)

def _now_epoch() -> int:
    return int(datetime.utcnow().timestamp())

# ---- KMS key/context fetch ----

async def fetch_key_context(session_id: Optional[str], request_id: Optional[str]) -> dict:
    """
    Busca contexto de chave no KMS. No seu fluxo atual, exige session_id.
    request_id é ignorado aqui (mantido por compatibilidade de assinatura).
    """
    if not session_id:
        raise ValueError("session_id is required to fetch key from KMS")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        url = f"{KMS_BASE_URL}/kms/get_key/{session_id}"
        resp = await client.get(url)

    if resp.status_code != 200:
        raise RuntimeError(f"KMS get_key failed ({resp.status_code}): {resp.text}")

    ctx = resp.json()
    exp = ctx.get("expires_at")
    now = _now_epoch()
    if isinstance(exp, int) and exp < now:
        raise RuntimeError("Session key expired")

    return ctx

# ---- Interceptor fetch ----

async def fetch_message_from_interceptor(request_id: str) -> dict:
    """Busca a mensagem original no Interceptor API pelo request_id"""
    if not request_id:
        raise ValueError("request_id required to fetch message from Interceptor")

    url = "http://localhost:8080/intercept/message"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(url, params={"request_id": request_id})

    if resp.status_code != 200:
        raise RuntimeError(f"Interceptor fetch failed ({resp.status_code}): {resp.text}")

    return resp.json()

# ---- AEAD operations ----

def _derive_aead_key(ctx: dict, algorithm: str, for_encrypt: bool) -> Tuple[bytes, int]:
    alg = algorithm.upper()
    if alg == "AES256_GCM":
        length = 32
        info = HKDF_INFO_ENCRYPT if for_encrypt else HKDF_INFO_DECRYPT
        return _hkdf_derive(ctx["key_material"], length, info), AESGCM_NONCE_SIZE
    elif alg == "CHACHA20_POLY1305":
        length = 32
        info = HKDF_INFO_ENCRYPT if for_encrypt else HKDF_INFO_DECRYPT
        return _hkdf_derive(ctx["key_material"], length, info), CHACHA20_NONCE_SIZE
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def aead_encrypt(ctx: dict, algorithm: str, plaintext: bytes, aad: Optional[bytes]) -> Tuple[str, str]:
    key, nonce_size = _derive_aead_key(ctx, algorithm, for_encrypt=True)
    nonce = os.urandom(nonce_size)
    aead = AESGCM(key) if algorithm.upper() == "AES256_GCM" else ChaCha20Poly1305(key)
    ciphertext = aead.encrypt(nonce, plaintext, aad)
    return b64e(nonce), b64e(ciphertext)

def aead_decrypt(ctx: dict, algorithm: str, nonce_b64: str, ciphertext_b64: str, aad: Optional[bytes]) -> bytes:
    key, _ = _derive_aead_key(ctx, algorithm, for_encrypt=False)
    nonce = b64d(nonce_b64)
    ciphertext = b64d(ciphertext_b64)
    aead = AESGCM(key) if algorithm.upper() == "AES256_GCM" else ChaCha20Poly1305(key)
    return aead.decrypt(nonce, ciphertext, aad)