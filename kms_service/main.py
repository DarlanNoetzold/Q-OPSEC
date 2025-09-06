from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import CreateKeyRequest, KeyResponse
from key_manager import build_session
from storage import save_session, get_session

app = FastAPI(title="KMS Service", version="2.0.0")

@app.post("/kms/create_key", response_model=KeyResponse)
async def create_key(req: CreateKeyRequest, db: Session = Depends(get_db)):
    session_id, alg, key_material, expires = build_session(req.session_id, req.algorithm, req.ttl_seconds)
    save_session(db, session_id, alg, key_material, expires)
    return KeyResponse(session_id=session_id, algorithm=alg,
                       key_material=key_material, expires_at=expires)

@app.get("/kms/get_key/{session_id}", response_model=KeyResponse)
async def get_key(session_id: str, db: Session = Depends(get_db)):
    sess = get_session(db, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found")
    return KeyResponse(session_id=sess.session_id, algorithm=sess.algorithm,
                       key_material=sess.key_material, expires_at=sess.expires_at)