from fastapi import FastAPI
from models import HandshakeRequest, HandshakeResponse
from negotiator import negotiate_algorithms, create_session

app = FastAPI(title="Handshake Negotiator", version="1.0.0")


@app.post("/handshake", response_model=HandshakeResponse)
async def handshake(req: HandshakeRequest):
    chosen, fallback_used = negotiate_algorithms(req.proposed, req.proposed)  # simplificado!
    session_id, expires = create_session(chosen)

    return HandshakeResponse(
        negotiated=chosen,
        session_key_id=session_id,
        expires_at=expires,
        fallback_used=fallback_used,
        message="Negotiation successful" if not fallback_used else "Negotiation with fallback"
    )