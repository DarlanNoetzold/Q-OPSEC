import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from models import NegotiationRequest, NegotiationResponse
from negotiator import negotiate_algorithms

app = FastAPI(title="Handshake Negotiator", version="2.1.0")

KMS_URL = "http://localhost:8002/kms/create_key"
KDE_URL = "http://localhost:8003/deliver"  # novo: chamar o KDE após o KMS

@app.post("/handshake", response_model=NegotiationResponse)
async def handshake(req: NegotiationRequest):
    # Requested = primeiro proposto (para exibir ao cliente)
    requested_alg = req.proposed[0] if req.proposed else "UNKNOWN"

    # Negocia internamente o alvo para enviar ao KMS
    chosen_alg, session_id, _, _ = negotiate_algorithms(req)

    # 1) KMS: criar chave/sessão
    async with httpx.AsyncClient() as client:
        kms_resp = await client.post(KMS_URL, json={
            "session_id": session_id,
            "algorithm": chosen_alg,
            "ttl_seconds": 300
        })

    if kms_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Erro ao criar chave no KMS: {kms_resp.text}")

    key_data = kms_resp.json()

    # Valores reais vindos do KMS
    actual_selected = key_data["selected_algorithm"]
    actual_fallback = key_data["fallback_applied"]
    actual_reason = key_data.get("fallback_reason")
    actual_source = key_data["source_of_key"]

    # 2) KDE: entregar a chave ao destino via API (padrão)
    # Se preferir, podemos tornar isso configurável via NegotiationRequest futuramente.
    kde_payload = {
        "session_id": key_data["session_id"],
        "destination": req.destination,
        "delivery_method": "API",
        "key_material": key_data["key_material"],
        "algorithm": actual_selected,
        "expires_at": key_data["expires_at"]
    }

    async with httpx.AsyncClient() as client:
        kde_resp = await client.post(KDE_URL, json=kde_payload)

    # Construir mensagem final ao cliente (não quebrar handshake se KDE falhar)
    if kde_resp.status_code == 200:
        kde_data = kde_resp.json()
        if actual_fallback:
            message = f"Negotiation completed with fallback: {requested_alg} -> {actual_selected}. Delivery: {kde_data.get('status', 'unknown')}"
        else:
            message = f"Negotiation completed successfully. Delivery: {kde_data.get('status', 'unknown')}"
    else:
        # Retornar sucesso do handshake, mas indicando falha de entrega
        if actual_fallback:
            message = f"Negotiation completed with fallback: {requested_alg} -> {actual_selected}. Delivery failed: {kde_resp.text[:200]}"
        else:
            message = f"Negotiation completed. Delivery failed: {kde_resp.text[:200]}"

    return NegotiationResponse(
        session_id=key_data["session_id"],
        requested_algorithm=requested_alg,   # o que foi pedido originalmente
        selected_algorithm=actual_selected,  # o que o KMS realmente usou
        key_material=key_data["key_material"],
        expires_at=key_data["expires_at"],
        fallback_applied=actual_fallback,    # fallback real do KMS
        fallback_reason=actual_reason,       # motivo real do KMS
        source_of_key=actual_source,         # fonte real do KMS
        message=message
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)