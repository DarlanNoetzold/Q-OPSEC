import httpx
import uvicorn
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from models import NegotiationRequest, NegotiationResponse
from negotiator import negotiate_algorithms
from urllib.parse import urlsplit, urlunsplit

app = FastAPI(title="Handshake Negotiator", version="2.2.0")

KMS_URL = "http://localhost:8002/kms/create_key"
KDE_URL = "http://localhost:8003/deliver"
CRYPTO_URL = "http://localhost:8004/encrypt/by-request-id"

def normalize_destination(dest: str) -> str:
    try:
        parts = urlsplit(dest)
        if parts.scheme not in ("http", "https"):
            return dest
        path = parts.path or ""
        if path.strip() in ("", "/"):
            path = "/receiver"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    except Exception:
        return dest

@app.post("/handshake", response_model=NegotiationResponse)
async def handshake(req: NegotiationRequest):
    request_id = req.request_id or f"req_{uuid4()}"

    requested_alg = req.proposed[0] if req.proposed else "UNKNOWN"
    chosen_alg, session_id, _, _ = negotiate_algorithms(req)

    # 1) KMS: criar/registrar chave para a sessão
    kms_payload = {
        "session_id": session_id,
        "request_id": request_id,
        "algorithm": chosen_alg,
        "ttl_seconds": 300
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        kms_resp = await client.post(KMS_URL, json=kms_payload)

    if kms_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Erro ao criar chave no KMS: {kms_resp.text}")

    try:
        key_data = kms_resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resposta inválida do KMS: {e}")

    actual_selected = key_data.get("selected_algorithm", chosen_alg)
    actual_fallback = key_data.get("fallback_applied", False)
    actual_reason = key_data.get("fallback_reason")
    actual_source = key_data.get("source_of_key", "unknown")

    message = (
        f"Negotiation completed with fallback: {requested_alg} -> {actual_selected}"
        if actual_fallback else
        "Negotiation completed successfully"
    )

    # 2) KDE: entregar chave/dados
    normalized_dest = normalize_destination(req.destination)
    delivery_payload = {
        "session_id": key_data["session_id"],
        "request_id": request_id,
        "destination": normalized_dest,
        "delivery_method": "API",
        "key_material": key_data["key_material"],
        "algorithm": actual_selected,
        "expires_at": key_data["expires_at"]
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        kde_resp = await client.post(KDE_URL, json=delivery_payload)

    kde_data: dict | str
    if kde_resp.status_code != 200:
        kde_data = {"error": f"HTTP {kde_resp.status_code}", "body": kde_resp.text}
    else:
        try:
            kde_data = kde_resp.json()
        except Exception:
            kde_data = {"raw": kde_resp.text}

    # 3) CRYPTO: solicitar criptografia da mensagem do Interceptor
    crypto_payload = {
        "request_id": request_id,
        "session_id": key_data["session_id"],
        "algorithm": actual_selected
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        crypto_resp = await client.post(CRYPTO_URL, json=crypto_payload)

    crypto_nonce_b64 = None
    crypto_ciphertext_b64 = None
    crypto_algorithm = None
    crypto_expires_at = None

    if crypto_resp.status_code == 200:
        try:
            c = crypto_resp.json()
            crypto_nonce_b64 = c.get("nonce_b64")
            crypto_ciphertext_b64 = c.get("ciphertext_b64")
            crypto_algorithm = c.get("algorithm")
            crypto_expires_at = c.get("expires_at")
        except Exception:
            # se parsing falhar, mantém None
            pass
    else:
        # opcional: incorporar status de erro do crypto no delivery_status
        pass

    return NegotiationResponse(
        request_id=request_id,
        session_id=key_data["session_id"],
        requested_algorithm=requested_alg,
        selected_algorithm=actual_selected,
        key_material=key_data["key_material"],
        expires_at=key_data["expires_at"],
        fallback_applied=actual_fallback,
        fallback_reason=actual_reason,
        source_of_key=actual_source,
        message=message,
        delivery_status=str({"kde": kde_data}),  # mantém KDE aqui
        crypto_nonce_b64=crypto_nonce_b64,
        crypto_ciphertext_b64=crypto_ciphertext_b64,
        crypto_algorithm=crypto_algorithm,
        crypto_expires_at=crypto_expires_at,
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)