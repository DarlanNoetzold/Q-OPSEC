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
    chosen_alg, session_id = negotiate_algorithms(req)

    async with httpx.AsyncClient() as client:
        resp = await client.post(KMS_URL, json={
            "session_id": session_id,
            "algorithm": chosen_alg,
            "ttl_seconds": 300
        })
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Erro ao criar chave no KMS")

    key_data = resp.json()

    return NegotiationResponse(
        negotiated=chosen_alg,
        session_id=key_data["session_id"],
        key_material=key_data["key_material"],
        expires_at=key_data["expires_at"],
        fallback_used=False,
        message="Negotiation concluída com sucesso"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)