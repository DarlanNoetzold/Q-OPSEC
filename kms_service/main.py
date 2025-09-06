from fastapi import FastAPI, HTTPException
import uvicorn
from models import CreateKeyRequest, KeyResponse
from key_manager import build_session
from storage import save_session, get_session

app = FastAPI(title="KMS Service", version="2.0.0")

@app.post("/kms/create_key", response_model=KeyResponse)
async def create_key(req: CreateKeyRequest):
    session_id, alg, key_material, expires = build_session(req.session_id, req.algorithm, req.ttl_seconds)
    await save_session(session_id, alg, key_material, expires)
    return KeyResponse(session_id=session_id, algorithm=alg, key_material=key_material, expires_at=expires)

@app.get("/kms/get_key/{session_id}", response_model=KeyResponse)
async def get_key(session_id: str):
    sess = await get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")
    return KeyResponse(**sess)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)