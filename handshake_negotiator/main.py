import httpx
import uvicorn
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from models import NegotiationRequest, NegotiationResponse
from negotiator import negotiate_algorithms
from urllib.parse import urlsplit, urlunsplit

app = FastAPI(title="Handshake Negotiator", version="2.1.3")

KMS_URL = "http://localhost:8002/kms/create_key"
KDE_URL = "http://localhost:8003/deliver"

def normalize_destination(dest: str) -> str:
    """
    Se o destino vier sem path (ex.: http://localhost:9000),
    adiciona '/receiver'. Se já tiver path, mantém.
    """
    try:
        parts = urlsplit(dest)
        # Precisa ser http(s)
        if parts.scheme not in ("http", "https"):
            return dest  # deixa para o KDE validar e retornar erro claro

        # Se não tiver path ou for apenas '/', adiciona '/receiver'
        path = parts.path or ""
        if path.strip() in ("", "/"):
            path = "/receiver"

        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    except Exception:
        # Em caso de string malformada, retorna como está (KDE retornará erro)
        return dest

@app.post("/handshake", response_model=NegotiationResponse)
async def handshake(req: NegotiationRequest):
    # Garante um request_id
    request_id = req.request_id or f"req_{uuid4()}"

    requested_alg = req.proposed[0] if req.proposed else "UNKNOWN"
    chosen_alg, session_id, _, _ = negotiate_algorithms(req)

    # 1) Chama o KMS com request_id
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

    # 2) Chama o KDE para entregar a chave (com request_id e destino normalizado)
    normalized_dest = normalize_destination(req.destination)

    delivery_payload = {
        "session_id": key_data["session_id"],
        "request_id": request_id,                   # obrigatório para o KDE
        "destination": normalized_dest,
        "delivery_method": "API",
        "key_material": key_data["key_material"],
        "algorithm": actual_selected,
        "expires_at": key_data["expires_at"]
        # "metadata": {...}  # opcional
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        kde_resp = await client.post(KDE_URL, json=delivery_payload)

    if kde_resp.status_code != 200:
        # Propaga com mais contexto
        raise HTTPException(
            status_code=500,
            detail=f"Erro no KDE (HTTP {kde_resp.status_code}): {kde_resp.text}"
        )

    try:
        kde_data = kde_resp.json()
    except Exception:
        kde_data = {"raw": kde_resp.text}

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
        delivery_status=str(kde_data)
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)