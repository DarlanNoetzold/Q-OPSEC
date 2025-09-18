# receiver.py
import json
import os
import base64
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

app = FastAPI(title="Mock Key Receiver & Decryptor", version="1.1.4")

# Arquivos de persistência
KEYS_FILE = "received_keys.json"
LOG_FILE = "crypto_operations.log"
MSG_LOG_FILE = "messages.log"

# HKDF (deve casar 100% com o crypto_engine/config)
# Se no Crypto for None, deixe None aqui; se for bytes, copie o mesmo valor.
HKDF_SALT = None  # alinhe com o Crypto
# Use o MESMO label que o Crypto usa na derivação para ENCRYPT, pois o ciphertext foi gerado com essa chave.
HKDF_INFO_DECRYPT = b"oraculumprisec:crypto:encrypt"

# ---------- Utils ----------
def b64d(data_b64: str) -> bytes:
    return base64.b64decode(data_b64.encode("utf-8"))

def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def log_operation(operation: str, data: dict):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        **data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def log_message(request_id: str, nonce_b64: str, ciphertext_b64: str, plaintext: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "nonce_b64": nonce_b64,
        "ciphertext_b64": ciphertext_b64,
        "plaintext": plaintext
    }
    with open(MSG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_key(request_id: str, key_data: dict):
    keys: Dict[str, Any] = {}
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                keys = json.load(f)
        except Exception:
            keys = {}
    keys[request_id] = {**key_data, "received_at": datetime.now().isoformat()}
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2, ensure_ascii=False)

def load_key(request_id: str) -> Optional[dict]:
    if not os.path.exists(KEYS_FILE):
        return None
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            keys = json.load(f)
        return keys.get(request_id)
    except Exception:
        return None

def _hkdf_derive(key_material_b64: str, length: int, info: bytes) -> bytes:
    key_bytes = b64d(key_material_b64)
    if not key_bytes:
        raise ValueError("Empty key_material")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=HKDF_SALT, info=info)
    return hkdf.derive(key_bytes)

def _default_aad(session_id: str, request_id: str, algorithm: str) -> bytes:
    return f"session:{session_id}|request:{request_id}|alg:{algorithm}".encode("utf-8")

def decrypt_message(key_material_b64: str, algorithm: str, nonce_b64: str, ciphertext_b64: str, aad: bytes) -> bytes:
    alg = algorithm.upper()
    if alg == "AES256_GCM":
        key = _hkdf_derive(key_material_b64, 32, HKDF_INFO_DECRYPT)
        aead = AESGCM(key)
    elif alg == "CHACHA20_POLY1305":
        key = _hkdf_derive(key_material_b64, 32, HKDF_INFO_DECRYPT)
        aead = ChaCha20Poly1305(key)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    nonce = b64d(nonce_b64)
    ciphertext = b64d(ciphertext_b64)
    return aead.decrypt(nonce, ciphertext, aad)

def _flatten_if_nested(data: Dict[str, Any]) -> Dict[str, Any]:
    # Tenta payloads aninhados comuns
    for key in ("payload", "data", "message", "record", "event"):
        if isinstance(data.get(key), dict):
            inner = data[key]
            has_crypto = any(k in inner for k in ("cryptoCiphertextB64", "ciphertext_b64", "ciphertext", "ciphertextB64"))
            has_req = any(k in inner for k in ("requestId", "request_id"))
            if has_crypto or has_req:
                return {**data, **inner}
    return data

# ---------- Model para /decrypt ----------
class DecryptRequest(BaseModel):
    request_id: str
    nonce_b64: str
    ciphertext_b64: str
    algorithm: str = "AES256_GCM"
    aad_b64: Optional[str] = None

# ---------- Endpoint combinado (/receiver) ----------
@app.post("/receiver")
async def receive(req: Request):
    """
    Recebe:
    - Entrega de chaves (KDE): contém "key_material" e "algorithm"
    - Payload criptografado (Validation Send API): contém cryptoCiphertextB64/cryptoNonceB64 (ou ciphertext_b64/nonce_b64)
    Também aceita payloads aninhados em {payload|data|message|record|event}.
    """
    try:
        raw = await req.json()
        log_operation("debug_received_payload", {"raw_payload": raw if isinstance(raw, dict) else str(raw)})
        data = _flatten_if_nested(raw)

        # Normalização de campos (camelCase e snake_case)
        request_id = data.get("request_id") or data.get("requestId")
        session_id = data.get("session_id") or data.get("sessionId")

        # Inclui selectedAlgorithm
        algorithm = (
            data.get("algorithm")
            or data.get("cryptoAlgorithm")
            or data.get("selectedAlgorithm")
            or "AES256_GCM"
        )

        key_material = data.get("key_material")  # Enviado pelo KDE

        # Criptograma (Validation)
        nonce_b64 = (
            data.get("cryptoNonceB64")
            or data.get("nonce_b64")
            or data.get("nonceB64")
            or data.get("nonce")
        )
        ciphertext_b64 = (
            data.get("cryptoCiphertextB64")
            or data.get("ciphertext_b64")
            or data.get("ciphertextB64")
            or data.get("ciphertext")
        )
        # Se vier o AAD explícito, usamos ele (recomendado para evitar divergência)
        aad_b64 = data.get("cryptoAadB64") or data.get("aad_b64") or data.get("aadB64")

        received_fields = list(data.keys())

        if not request_id:
            detail = {"error": "request_id/requestId is required", "received_fields": received_fields}
            log_operation("receiver_error", detail)
            raise HTTPException(status_code=400, detail=detail)

        # 1) Entrega de chave do KDE
        if key_material:
            save_key(request_id, data)
            log_operation("key_received", {
                "request_id": request_id,
                "session_id": session_id,
                "algorithm": algorithm,
                "expires_at": data.get("expires_at") or data.get("cryptoExpiresAt"),
                "has_key_material": True
            })
            return {
                "ok": True,
                "type": "key_delivery",
                "message": f"Key for request_id {request_id} saved successfully",
                "received_fields": received_fields
            }

        # 2) Payload criptografado do Validation
        if nonce_b64 and ciphertext_b64:
            key_data = load_key(request_id)
            if not key_data:
                detail = {"error": f"Key not found for request_id: {request_id}", "received_fields": received_fields}
                log_operation("receiver_error", detail)
                raise HTTPException(status_code=404, detail=detail)

            key_material_saved = key_data.get("key_material")
            if not key_material_saved:
                detail = {"error": "Key material not available", "request_id": request_id}
                log_operation("receiver_error", detail)
                raise HTTPException(status_code=400, detail=detail)

            alg = algorithm or key_data.get("algorithm") or "AES256_GCM"
            sid = session_id or key_data.get("session_id") or ""

            # Use o AAD que veio do Validation (preferível). Se não vier, gera o padrão.
            aad = b64d(aad_b64) if aad_b64 else _default_aad(sid, request_id, alg)

            try:
                plaintext_bytes = decrypt_message(
                    key_material_b64=key_material_saved,
                    algorithm=alg,
                    nonce_b64=nonce_b64,
                    ciphertext_b64=ciphertext_b64,
                    aad=aad
                )
                plaintext = plaintext_bytes.decode("utf-8")

                log_operation("message_decrypted", {
                    "request_id": request_id,
                    "session_id": sid,
                    "algorithm": alg,
                    "nonce_b64": nonce_b64,
                    "ciphertext_b64": ciphertext_b64,
                    "plaintext": plaintext,
                    "plaintext_length": len(plaintext)
                })
                log_message(request_id, nonce_b64, ciphertext_b64, plaintext)

                return {
                    "ok": True,
                    "type": "encrypted_payload",
                    "request_id": request_id,
                    "session_id": sid,
                    "algorithm": alg,
                    "plaintext": plaintext,
                    "decrypted_at": datetime.now().isoformat()
                }
            except InvalidTag:
                # Chave/AAD/nonce/ciphertext divergentes
                log_operation("decrypt_error", {
                    "request_id": request_id,
                    "error": "InvalidTag (provável mismatch de AAD/HKDF salt/info/chave)",
                    "nonce_b64": nonce_b64,
                    "ciphertext_b64": ciphertext_b64,
                    "algorithm": alg
                })
                raise HTTPException(status_code=400, detail="Decryption failed: InvalidTag")
            except Exception as decrypt_err:
                log_operation("decrypt_error", {
                    "request_id": request_id,
                    "error": str(decrypt_err),
                    "nonce_b64": nonce_b64,
                    "ciphertext_b64": ciphertext_b64,
                    "algorithm": alg
                })
                raise HTTPException(status_code=400, detail=f"Decryption failed: {decrypt_err}")

        # Nada reconhecido
        detail = {
            "error": "Payload not recognized: expected key_material or crypto{Nonce,Ciphertext}B64",
            "received_fields": received_fields,
            "tip": "verifique se Validation está repassando cryptoNonceB64/cryptoCiphertextB64 (e cryptoAadB64 se possível)"
        }
        log_operation("receiver_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    except HTTPException:
        raise
    except Exception as e:
        # Fallback melhorado
        error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
        try:
            received_fields = list(raw.keys()) if 'raw' in locals() else []
        except Exception:
            received_fields = []
        detail = {"error": f"Error processing payload: {error_msg}", "received_fields": received_fields}
        log_operation("receiver_error", detail)
        raise HTTPException(status_code=400, detail=detail)

# ---------- Endpoint de decrypt direto ----------
@app.post("/decrypt")
async def decrypt_message_endpoint(req: DecryptRequest):
    try:
        key_data = load_key(req.request_id)
        if not key_data:
            raise HTTPException(status_code=404, detail=f"Key not found for request_id: {req.request_id}")

        key_material = key_data.get("key_material")
        session_id = key_data.get("session_id", "")
        if not key_material:
            raise HTTPException(status_code=400, detail="Key material not available")

        aad = b64d(req.aad_b64) if req.aad_b64 else _default_aad(session_id, req.request_id, req.algorithm)

        plaintext_bytes = decrypt_message(
            key_material_b64=key_material,
            algorithm=req.algorithm,
            nonce_b64=req.nonce_b64,
            ciphertext_b64=req.ciphertext_b64,
            aad=aad
        )
        plaintext = plaintext_bytes.decode("utf-8")

        log_operation("message_decrypted", {
            "request_id": req.request_id,
            "session_id": session_id,
            "algorithm": req.algorithm,
            "nonce_b64": req.nonce_b64,
            "ciphertext_b64": req.ciphertext_b64,
            "plaintext": plaintext,
            "plaintext_length": len(plaintext)
        })
        log_message(req.request_id, req.nonce_b64, req.ciphertext_b64, plaintext)

        return {
            "request_id": req.request_id,
            "session_id": session_id,
            "algorithm": req.algorithm,
            "plaintext": plaintext,
            "decrypted_at": datetime.now().isoformat()
        }

    except InvalidTag:
        log_operation("decrypt_error", {"request_id": req.request_id, "error": "InvalidTag"})
        raise HTTPException(status_code=400, detail="Decryption failed: InvalidTag")
    except ValueError as e:
        log_operation("decrypt_error", {"request_id": req.request_id, "error": f"Decryption failed: {e}"})
        raise HTTPException(status_code=400, detail=f"Decryption error: {e}")
    except Exception as e:
        log_operation("decrypt_error", {"request_id": req.request_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# ---------- Auxiliares de inspeção ----------
@app.get("/keys")
async def list_saved_keys():
    if not os.path.exists(KEYS_FILE):
        return {"keys": {}}
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            keys = json.load(f)
        safe_keys = {req_id: {k: v for k, v in data.items() if k != "key_material"} for req_id, data in keys.items()}
        return {"keys": safe_keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading keys: {e}")

@app.get("/logs")
async def get_recent_logs(limit: int = 10):
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    try:
        logs = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line.strip()))
        return {"logs": logs[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")

@app.get("/messages")
async def get_messages(limit: int = 10):
    if not os.path.exists(MSG_LOG_FILE):
        return {"messages": []}
    try:
        msgs = []
        with open(MSG_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    msgs.append(json.loads(line.strip()))
        return {"messages": msgs[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading messages: {e}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "receiver",
        "version": "1.1.4",
        "files": {
            "keys_file_exists": os.path.exists(KEYS_FILE),
            "log_file_exists": os.path.exists(LOG_FILE),
            "messages_file_exists": os.path.exists(MSG_LOG_FILE)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("receiver:app", host="0.0.0.0", port=9000, reload=True)