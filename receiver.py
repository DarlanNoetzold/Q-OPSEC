# receiver.py
import json
import os
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

app = FastAPI(title="Mock Key Receiver & Decryptor", version="1.1.0")

# Arquivos de persistência
KEYS_FILE = "received_keys.json"
LOG_FILE = "crypto_operations.log"
MSG_LOG_FILE = "messages.log"

# Configurações HKDF (devem ser iguais ao crypto_engine)
HKDF_SALT = None  # ou configure igual ao crypto_engine
HKDF_INFO_DECRYPT = b"oraculumprisec:crypto:decrypt"

# ---------- Utils ----------
def b64d(data_b64: str) -> bytes:
    return base64.b64decode(data_b64.encode("utf-8"))

def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def log_operation(operation: str, data: dict):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
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

    keys[request_id] = {
        **key_data,
        "received_at": datetime.now().isoformat()
    }

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
    s = f"session:{session_id}|request:{request_id}|alg:{algorithm}"
    return s.encode("utf-8")

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

# ---------- Model para /decrypt direto ----------
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
    """
    try:
        data = await req.json()

        # Normalizar campos para suportar camelCase e snake_case
        request_id = data.get("request_id") or data.get("requestId")
        session_id = data.get("session_id") or data.get("sessionId")
        algorithm = data.get("algorithm") or data.get("cryptoAlgorithm") or "AES256_GCM"
        key_material = data.get("key_material")  # KDE usa "key_material"

        # Campos para mensagem criptografada (Validation Send API)
        nonce_b64 = data.get("cryptoNonceB64") or data.get("nonce_b64")
        ciphertext_b64 = data.get("cryptoCiphertextB64") or data.get("ciphertext_b64")

        if not request_id:
            raise HTTPException(status_code=400, detail="request_id/requestId is required")

        # Caso 1: entrega de chave do KDE
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
                "received_fields": list(data.keys())
            }

        # Caso 2: payload criptografado vindo do Validation Send API
        if nonce_b64 and ciphertext_b64:
            key_data = load_key(request_id)
            if not key_data:
                raise HTTPException(status_code=404, detail=f"Key not found for request_id: {request_id}")

            key_material = key_data.get("key_material")
            if not key_material:
                raise HTTPException(status_code=400, detail="Key material not available")

            # algoritmo: prioriza o do payload; senão, o salvo junto com a chave
            algorithm = algorithm or key_data.get("algorithm") or "AES256_GCM"
            sid = session_id or key_data.get("session_id") or ""

            aad = _default_aad(sid, request_id, algorithm)

            plaintext_bytes = decrypt_message(
                key_material=key_material,
                algorithm=algorithm,
                nonce_b64=nonce_b64,
                ciphertext_b64=ciphertext_b64,
                aad=aad
            )
            plaintext = plaintext_bytes.decode("utf-8")

            # Logs
            log_operation("message_decrypted", {
                "request_id": request_id,
                "session_id": sid,
                "algorithm": algorithm,
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
                "algorithm": algorithm,
                "plaintext": plaintext,
                "decrypted_at": datetime.now().isoformat()
            }

        # Nada reconhecido
        return {
            "ok": False,
            "error": "Payload not recognized: expected key_material or crypto{Nonce,Ciphertext}B64"
        }

    except HTTPException:
        raise
    except Exception as e:
        log_operation("receiver_error", {"error": str(e)})
        raise HTTPException(status_code=400, detail=f"Error processing payload: {e}")

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
            key_material,
            req.algorithm,
            req.nonce_b64,
            req.ciphertext_b64,
            aad
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
        safe_keys = {}
        for req_id, data in keys.items():
            safe_keys[req_id] = {k: v for k, v in data.items() if k != "key_material"}
        return {"keys": safe_keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading keys: {e}")

@app.get("/logs")
async def get_recent_logs(limit: int = 10):
    """
    Mostra os últimos N logs, incluindo:
    - key_received (como no exemplo que você mandou)
    - message_decrypted (com plaintext)
    - errors
    """
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
    """Lista mensagens descriptografadas (de messages.log)"""
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
        "version": "1.1.0",
        "files": {
            "keys_file_exists": os.path.exists(KEYS_FILE),
            "log_file_exists": os.path.exists(LOG_FILE),
            "messages_file_exists": os.path.exists(MSG_LOG_FILE)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("receiver:app", host="0.0.0.0", port=9000, reload=True)