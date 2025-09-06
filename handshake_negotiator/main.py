from fastapi import FastAPI
import uvicorn
from models import HandshakeRequest, HandshakeResponse
from negotiator import negotiate_algorithms, create_session

app = FastAPI(
    title="Handshake Negotiator",
    description="Negociador de protocolos (QKD / PQC / Clássicos / Híbridos)",
    version="2.0.0"
)

@app.post("/handshake", response_model=HandshakeResponse)
async def handshake(req: HandshakeRequest):
    chosen, fallback_used = negotiate_algorithms(req.proposed, req.proposed)
    session_id, expires = create_session(chosen)

    return HandshakeResponse(
        negotiated=chosen,
        session_key_id=session_id,
        expires_at=expires,
        fallback_used=fallback_used,
        message="Negotiation OK" if not fallback_used else "Negotiation with fallback"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)