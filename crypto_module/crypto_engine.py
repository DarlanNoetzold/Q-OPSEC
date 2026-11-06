import base64
import os
import time
import httpx
from datetime import datetime
from typing import Optional, Tuple

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from common.ttl_cache import TTLCache

from config import (
    KMS_BASE_URL,
    HTTP_TIMEOUT,
    AESGCM_NONCE_SIZE,
    CHACHA20_NONCE_SIZE,
    HKDF_SALT,
    HKDF_INFO_ENCRYPT,
    HKDF_INFO_DECRYPT,
)

_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
_KEY_CONTEXT_CACHE = TTLCache(max_entries=512)


def get_http_client() -> httpx.AsyncClient:
    """Reuse one connection pool instead of opening a client per request."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    return _HTTP_CLIENT


async def close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None and not _HTTP_CLIENT.is_closed:
        await _HTTP_CLIENT.aclose()
    _HTTP_CLIENT = None


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
    return int(time.time())


async def fetch_key_context(session_id: Optional[str], request_id: Optional[str]) -> dict:
    if not session_id:
        raise ValueError("session_id is required to fetch key from KMS")

    cached = _KEY_CONTEXT_CACHE.get(session_id)
    if cached is not None:
        return cached

    client = get_http_client()
    url = f"{KMS_BASE_URL}/kms/get_key/{session_id}"
    resp = await client.get(url)

    if resp.status_code != 200:
        raise RuntimeError(f"KMS get_key failed ({resp.status_code}): {resp.text}")

    ctx = resp.json()
    exp = ctx.get("expires_at")
    now = _now_epoch()
    if isinstance(exp, int) and exp < now:
        raise RuntimeError("Session key expired")

    ttl = exp - now if isinstance(exp, int) else 30
    if ttl > 0:
        _KEY_CONTEXT_CACHE.set(session_id, ctx, min(ttl, 300))
    return ctx


async def fetch_message_from_interceptor(request_id: str) -> dict:
    if not request_id:
        raise ValueError("request_id required to fetch message from Interceptor")

    url = "http://localhost:8080/intercept/message"
    resp = await get_http_client().get(url, params={"request_id": request_id})

    if resp.status_code != 200:
        raise RuntimeError(f"Interceptor fetch failed ({resp.status_code}): {resp.text}")

    return resp.json()


def _derive_aead_key(ctx: dict, algorithm: str, for_encrypt: bool) -> Tuple[bytes, int]:
    alg = algorithm.upper()
    # PQC/QKD algorithms negotiate the key-exchange/signature layer. Payload
    # encryption remains AEAD; KMS already returns the derived session key.
    if alg not in ("AES256_GCM", "CHACHA20_POLY1305"):
        alg = "AES256_GCM"
    if alg == "AES256_GCM":
        length = 32
        return _hkdf_derive(ctx["key_material"], length, HKDF_INFO_ENCRYPT), AESGCM_NONCE_SIZE
    elif alg == "CHACHA20_POLY1305":
        length = 32
        return _hkdf_derive(ctx["key_material"], length, HKDF_INFO_ENCRYPT), CHACHA20_NONCE_SIZE
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def aead_encrypt(ctx: dict, algorithm: str, plaintext: bytes, aad: Optional[bytes]) -> Tuple[str, str]:
    key, nonce_size = _derive_aead_key(ctx, algorithm, for_encrypt=True)
    nonce = os.urandom(nonce_size)
    aead = AESGCM(key) if algorithm.upper() not in ("CHACHA20_POLY1305",) else ChaCha20Poly1305(key)
    ciphertext = aead.encrypt(nonce, plaintext, aad)
    return b64e(nonce), b64e(ciphertext)

def aead_decrypt(ctx: dict, algorithm: str, nonce_b64: str, ciphertext_b64: str, aad: Optional[bytes]) -> bytes:
    key, _ = _derive_aead_key(ctx, algorithm, for_encrypt=False)
    nonce = b64d(nonce_b64)
    ciphertext = b64d(ciphertext_b64)
    aead = AESGCM(key) if algorithm.upper() not in ("CHACHA20_POLY1305",) else ChaCha20Poly1305(key)
    return aead.decrypt(nonce, ciphertext, aad)