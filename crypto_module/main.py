import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

from models import EncryptRequest, EncryptResponse, DecryptRequest, DecryptResponse, ErrorResponse
from crypto_engine import (
    fetch_key_context,
    fetch_message_from_interceptor,
    aead_encrypt,
    aead_decrypt,
    b64d,
    b64e,
)
from config import HOST, PORT

def _to_unix_ts(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            raise ValueError(f"Invalid expires_at format: {value}")
    if isinstance(value, datetime):
        return int(value.timestamp())
    return int(value)

app = FastAPI(
    title="OraculumPrisec Crypto Module",
    description="Encrypt/Decrypt service using session keys from KMS and message from Interceptor API",
    version="1.3.1",
)

def _default_aad(session_id: str, request_id: str, algorithm: str) -> bytes:
    s = f"session:{session_id}|request:{request_id}|alg:{algorithm}"
    return s.encode("utf-8")

@app.post("/encrypt", response_model=EncryptResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def encrypt(req: EncryptRequest):
    try:
        if not req.session_id:
            raise ValueError("session_id is required to fetch key from KMS")

        ctx = await fetch_key_context(session_id=req.session_id, request_id=None)
        session_id = ctx["session_id"]
        req_id = req.request_id or ctx.get("request_id") or ""

        if not req.plaintext_b64:
            fetch_flag = getattr(req, "fetch_from_interceptor", True)
            if not fetch_flag:
                raise ValueError("plaintext_b64 is required when fetch_from_interceptor=False")
            if not req_id:
                raise ValueError("request_id is required to fetch message from Interceptor")

            intercepted = await fetch_message_from_interceptor(req_id)
            msg = intercepted.get("message")
            if msg is None:
                raise ValueError("Interceptor response missing 'message' field")
            payload_b64 = b64e(msg.encode("utf-8"))
        else:
            payload_b64 = req.plaintext_b64

        plaintext = b64d(payload_b64) or b""

        aad = b64d(req.aad_b64) if req.aad_b64 else _default_aad(session_id, req_id, req.algorithm)

        nonce_b64, ciphertext_b64 = aead_encrypt(ctx, req.algorithm, plaintext, aad)

        return EncryptResponse(
            session_id=session_id,
            algorithm=req.algorithm,
            nonce_b64=nonce_b64,
            ciphertext_b64=ciphertext_b64,
            expires_at=_to_unix_ts(ctx["expires_at"]),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encrypt error: {e}")

# POST simples: request_id + session_id
class EncryptByRequestId(BaseModel):
    request_id: str = Field(..., description="Usado para buscar mensagem no Interceptor")
    session_id: str = Field(..., description="Usado para buscar chave no KMS")
    algorithm: str = Field("AES256_GCM", description="AES256_GCM | CHACHA20_POLY1305")
    aad_b64: str | None = None

@app.post("/encrypt/by-request-id", response_model=EncryptResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def encrypt_by_request_id(body: EncryptByRequestId):
    try:
        intercepted = await fetch_message_from_interceptor(body.request_id)
        msg = intercepted.get("message")
        if msg is None:
            raise ValueError("Interceptor response missing 'message' field")

        ctx = await fetch_key_context(session_id=body.session_id, request_id=None)

        aad = b64d(body.aad_b64) if body.aad_b64 else _default_aad(body.session_id, body.request_id, body.algorithm)

        nonce_b64, ciphertext_b64 = aead_encrypt(
            ctx,
            body.algorithm,
            msg.encode("utf-8"),
            aad,
        )

        return EncryptResponse(
            session_id=ctx["session_id"],
            algorithm=body.algorithm,
            nonce_b64=nonce_b64,
            ciphertext_b64=ciphertext_b64,
            expires_at=_to_unix_ts(ctx["expires_at"]),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encrypt error: {e}")

@app.post("/decrypt", response_model=DecryptResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def decrypt(req: DecryptRequest):
    try:
        if not req.session_id:
            raise ValueError("session_id is required to fetch key from KMS")

        ctx = await fetch_key_context(session_id=req.session_id, request_id=None)
        session_id = ctx["session_id"]
        req_id = req.request_id or ctx.get("request_id") or ""

        aad = b64d(req.aad_b64) if req.aad_b64 else _default_aad(session_id, req_id, req.algorithm)

        plaintext = aead_decrypt(ctx, req.algorithm, req.nonce_b64, req.ciphertext_b64, aad)
        return DecryptResponse(
            session_id=session_id,
            algorithm=req.algorithm,
            plaintext_b64=b64e(plaintext),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decrypt error: {e}")

@app.get("/health")
async def health():
    return {"status": "healthy", "module": "crypto", "version": "1.3.1"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)