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
    CONTEXT_API_BASE_URL,
    HTTP_TIMEOUT,
    CONTEXT_TIMEOUT,
    AESGCM_NONCE_SIZE,
    CHACHA20_NONCE_SIZE,
    HKDF_SALT,
    HKDF_INFO_ENCRYPT,
    HKDF_INFO_DECRYPT,
)

# ---------- Helpers ----------

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

async def fetch_key_context(session_id: Optional[str], request_id: Optional[str]) -> dict:
    if not session_id and not request_id:
        raise ValueError("Either session_id or request_id must be provided")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        if session_id:
            url = f"{KMS_BASE_URL}/kms/get_key/{session_id}"
            resp = await client.get(url)
        else:
            url = f"{KMS_BASE_URL}/kms/get_key"
            resp = await client.get(url, params={"request_id": request_id})

    if resp.status_code != 200:
        raise RuntimeError(f"KMS get_key failed ({resp.status_code}): {resp.text}")

    ctx = resp.json()
    exp = ctx.get("expires_at")
    now = _now_epoch()
    if isinstance(exp, int) and exp < now:
        raise RuntimeError("Session key expired")

    return ctx

async def fetch_payload_from_context(request_id: str) -> str:
    """Busca o payload (base64) no Context API via request_id"""
    if not request_id:
        raise ValueError("request_id required to fetch payload from Context API")

    url = f"{CONTEXT_API_BASE_URL}/context/payload"
    async with httpx.AsyncClient(timeout=CONTEXT_TIMEOUT) as client:
        resp = await client.get(url, params={"request_id": request_id})

    if resp.status_code != 200:
        raise RuntimeError(f"Context API fetch failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    payload_b64 = data.get("payload_b64")
    if not payload_b64:
        raise RuntimeError("Context API response missing payload_b64")

    return payload_b64

# ---------- AEAD operations ----------

def _derive_aead_key(ctx: dict, algorithm: str, for_encrypt: bool) -> Tuple[bytes, int]:
    if algorithm.upper() == "AES256_GCM":
        length = 32
        info = HKDF_INFO_ENCRYPT if for_encrypt else HKDF_INFO_DECRYPT
        return _hkdf_derive(ctx["key_material"], length, info), AESGCM_NONCE_SIZE
    elif algorithm.upper() == "CHACHA20_POLY1305":
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