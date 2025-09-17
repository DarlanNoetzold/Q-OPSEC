# receiver.py
import json
import os
import base64
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

app = FastAPI(title="Mock Key Receiver & Decryptor", version="1.0.0")

# Arquivos de persistência
KEYS_FILE = "received_keys.json"
LOG_FILE = "crypto_operations.log"

# Configurações HKDF (devem ser iguais ao crypto_engine)
HKDF_SALT = None  # ou configure igual ao crypto_engine
HKDF_INFO_DECRYPT = b"oraculumprisec:crypto:decrypt"

class DecryptRequest(BaseModel):
    request_id: str
    nonce_b64: str
    ciphertext_b64: str
    algorithm: str = "AES256_GCM"
    aad_b64: Optional[str] = None

def log_operation(operation: str, data: dict):
    """Log das operações em arquivo"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "operation": operation,
        **data
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def save_key(request_id: str, key_data: dict):
    """Salva chave recebida em arquivo JSON"""
    keys = {}
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                keys = json.load(f)
        except:
            keys = {}
    
    keys[request_id] = {
        **key_data,
        "received_at": datetime.now().isoformat()
    }
    
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2, ensure_ascii=False)

def load_key(request_id: str) -> Optional[dict]:
    """Carrega chave salva pelo request_id"""
    if not os.path.exists(KEYS_FILE):
        return None
    
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            keys = json.load(f)
        return keys.get(request_id)
    except:
        return None

def b64d(data_b64: str) -> bytes:
    """Decode base64"""
    return base64.b64decode(data_b64.encode("utf-8"))

def _hkdf_derive(key_material_b64: str, length: int, info: bytes) -> bytes:
    """Deriva chave usando HKDF (igual ao crypto_engine)"""
    key_bytes = b64d(key_material_b64)
    if not key_bytes:
        raise ValueError("Empty key_material")
    
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=HKDF_SALT, info=info)
    return hkdf.derive(key_bytes)

def _default_aad(session_id: str, request_id: str, algorithm: str) -> bytes:
    """AAD padrão (igual ao crypto_engine)"""
    s = f"session:{session_id}|request:{request_id}|alg:{algorithm}"
    return s.encode("utf-8")

def decrypt_message(key_material_b64: str, algorithm: str, nonce_b64: str, 
                   ciphertext_b64: str, aad: bytes) -> bytes:
    """Descriptografa mensagem usando AEAD"""
    alg = algorithm.upper()
    
    if alg == "AES256_GCM":
        length = 32
        key = _hkdf_derive(key_material_b64, length, HKDF_INFO_DECRYPT)
        aead = AESGCM(key)
    elif alg == "CHACHA20_POLY1305":
        length = 32
        key = _hkdf_derive(key_material_b64, length, HKDF_INFO_DECRYPT)
        aead = ChaCha20Poly1305(key)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    nonce = b64d(nonce_b64)
    ciphertext = b64d(ciphertext_b64)
    
    return aead.decrypt(nonce, ciphertext, aad)

@app.post("/receiver")
async def receive_key(req: Request):
    """Endpoint original - recebe e salva chaves do KDE"""
    try:
        data = await req.json()
        
        # Extrai informações principais
        request_id = data.get("request_id")
        if not request_id:
            return {"error": "request_id is required"}
        
        # Salva a chave completa
        save_key(request_id, data)
        
        # Log da operação
        log_operation("key_received", {
            "request_id": request_id,
            "session_id": data.get("session_id"),
            "algorithm": data.get("algorithm"),
            "expires_at": data.get("expires_at"),
            "has_key_material": bool(data.get("key_material"))
        })
        
        return {
            "ok": True, 
            "message": f"Key for request_id {request_id} saved successfully",
            "received_fields": list(data.keys())
        }
        
    except Exception as e:
        log_operation("key_receive_error", {"error": str(e)})
        raise HTTPException(status_code=400, detail=f"Error processing key: {e}")

@app.post("/decrypt")
async def decrypt_message_endpoint(req: DecryptRequest):
    """Novo endpoint - descriptografa mensagem usando chave salva"""
    try:
        # Carrega chave salva
        key_data = load_key(req.request_id)
        if not key_data:
            raise HTTPException(status_code=404, detail=f"Key not found for request_id: {req.request_id}")
        
        key_material = key_data.get("key_material")
        session_id = key_data.get("session_id", "")
        
        if not key_material:
            raise HTTPException(status_code=400, detail="Key material not available")
        
        # AAD: usa o fornecido ou gera padrão
        if req.aad_b64:
            aad = b64d(req.aad_b64)
        else:
            aad = _default_aad(session_id, req.request_id, req.algorithm)
        
        # Descriptografa
        plaintext_bytes = decrypt_message(
            key_material, 
            req.algorithm, 
            req.nonce_b64, 
            req.ciphertext_b64, 
            aad
        )
        
        plaintext = plaintext_bytes.decode("utf-8")
        
        # Log completo da operação
        log_operation("message_decrypted", {
            "request_id": req.request_id,
            "session_id": session_id,
            "algorithm": req.algorithm,
            "nonce_b64": req.nonce_b64,
            "ciphertext_b64": req.ciphertext_b64,
            "plaintext": plaintext,
            "plaintext_length": len(plaintext)
        })
        
        return {
            "request_id": req.request_id,
            "session_id": session_id,
            "algorithm": req.algorithm,
            "plaintext": plaintext,
            "decrypted_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        log_operation("decrypt_error", {
            "request_id": req.request_id,
            "error": f"Decryption failed: {e}"
        })
        raise HTTPException(status_code=400, detail=f"Decryption error: {e}")
    except Exception as e:
        log_operation("decrypt_error", {
            "request_id": req.request_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/keys")
async def list_saved_keys():
    """Endpoint para listar chaves salvas"""
    if not os.path.exists(KEYS_FILE):
        return {"keys": {}}
    
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            keys = json.load(f)
        
        # Remove key_material da resposta por segurança
        safe_keys = {}
        for req_id, data in keys.items():
            safe_keys[req_id] = {
                k: v for k, v in data.items() 
                if k != "key_material"
            }
        
        return {"keys": safe_keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading keys: {e}")

@app.get("/logs")
async def get_recent_logs(limit: int = 10):
    """Endpoint para ver logs recentes"""
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    
    try:
        logs = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line.strip()))
        
        # Retorna os últimos N logs
        return {"logs": logs[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "receiver", 
        "version": "1.0.0",
        "files": {
            "keys_file_exists": os.path.exists(KEYS_FILE),
            "log_file_exists": os.path.exists(LOG_FILE)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("receiver:app", host="0.0.0.0", port=9000, reload=True)