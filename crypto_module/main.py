import uvicorn
from fastapi import FastAPI, HTTPException
from models import EncryptRequest, EncryptResponse, DecryptRequest, DecryptResponse, ErrorResponse
from crypto_engine import (
    fetch_key_context,
    fetch_payload_from_context,
    aead_encrypt,
    aead_decrypt,
    b64d,
    b64e,
)
from config import HOST, PORT
from datetime import datetime

def _to_unix_ts(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            # tenta parse ISO 8601
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            raise ValueError(f"Invalid expires_at format: {value}")
    if isinstance(value, datetime):
        return int(value.timestamp())
    return int(value)

app = FastAPI(
    title="OraculumPrisec Crypto Module",
    description="Encrypt/Decrypt service using session keys from KMS and payload from Context API",
    version="1.1.0",
)

def _default_aad(session_id: str, request_id: str, algorithm: str) -> bytes:
    # AAD padrão vinculando criptografia ao contexto
    s = f"session:{session_id}|request:{request_id}|alg:{algorithm}"
    return s.encode("utf-8")

@app.post("/encrypt", response_model=EncryptResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def encrypt(req: EncryptRequest):
    try:
        # Contexto da chave
        ctx = await fetch_key_context(req.session_id, req.request_id)
        session_id = ctx["session_id"]
        req_id = req.request_id or ctx.get("request_id") or ""

        # Payload: se não veio inline e fetch_from_context=True, busca no Context API
        if not req.plaintext_b64:
            if not req.fetch_from_context:
                raise ValueError("plaintext_b64 is required when fetch_from_context=False")
            if not req_id:
                raise ValueError("request_id is required to fetch payload from Context API")
            payload_b64 = await fetch_payload_from_context(req_id)
        else:
            payload_b64 = req.plaintext_b64

        plaintext = b64d(payload_b64) or b""

        # AAD
        aad = b64d(req.aad_b64) if req.aad_b64 else _default_aad(session_id, req_id, req.algorithm)

        # Encrypt
        nonce_b64, ciphertext_b64 = aead_encrypt(ctx, req.algorithm, plaintext, aad)
        return EncryptResponse(
            session_id=session_id,
            algorithm=req.algorithm,
            nonce_b64=nonce_b64,
            ciphertext_b64=ciphertext_b64,
            expires_at=_to_unix_ts(ctx["expires_at"]),  # <-- normaliza
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
        ctx = await fetch_key_context(req.session_id, req.request_id)
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
    return {"status": "healthy", "module": "crypto", "version": "1.1.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)