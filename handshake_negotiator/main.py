import uuid
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from models import NegotiationRequest, NegotiationResponse
from negotiator import negotiate_algorithms

app = FastAPI(title="Handshake Negotiator", version="2.0.0")

KMS_URL = "http://localhost:8002/kms/create_key"


@app.post("/handshake", response_model=NegotiationResponse)
async def handshake(req: NegotiationRequest):
    requested_alg = req.proposed[0] if req.proposed else "UNKNOWN"

    chosen_alg, session_id, _, _ = negotiate_algorithms(req)

    # Call KMS to create the key
    async with httpx.AsyncClient() as client:
        resp = await client.post(KMS_URL, json={
            "session_id": session_id,
            "algorithm": chosen_alg,
            "ttl_seconds": 300
        })

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Erro ao criar chave no KMS")

    key_data = resp.json()

    actual_selected = key_data["selected_algorithm"]
    actual_fallback = key_data["fallback_applied"]
    actual_reason = key_data.get("fallback_reason")
    actual_source = key_data["source_of_key"]

    if actual_fallback:
        message = f"Negotiation completed with fallback: {requested_alg} -> {actual_selected}"
    else:
        message = "Negotiation completed successfully"

    return NegotiationResponse(
        session_id=key_data["session_id"],
        requested_algorithm=requested_alg,  # O que foi pedido originalmente
        selected_algorithm=actual_selected,  # O que o KMS realmente usou
        key_material=key_data["key_material"],
        expires_at=key_data["expires_at"],
        fallback_applied=actual_fallback,  # Fallback real do KMS
        fallback_reason=actual_reason,  # Motivo real do KMS
        source_of_key=actual_source,  # Fonte real do KMS
        message=message
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)