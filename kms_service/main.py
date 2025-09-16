from fastapi import FastAPI, HTTPException, Response
import uvicorn
from models import CreateKeyRequest, CreateKeyResponse, KeyResponse
from key_manager import build_session, get_supported_algorithms, get_algorithm_info
from storage import save_session, get_session, get_session_by_request
from datetime import datetime

app = FastAPI(title="KMS Service", version="2.0.0")


@app.get("/kms/supported_algorithms")
def supported_algorithms():
    return get_supported_algorithms()


@app.get("/kms/algorithm_info/{algorithm}")
def algorithm_info(algorithm: str):
    return get_algorithm_info(algorithm)


def _to_unix_ts(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    return int(value)


@app.post("/kms/create_key", response_model=CreateKeyResponse)
async def create_key(req: CreateKeyRequest):
    session_id, request_id, selected_alg, key_material, expires_at, fallback_applied, fallback_reason, source_of_key = build_session(
        req.session_id,
        req.request_id,
        req.algorithm,
        req.ttl_seconds
    )

    await save_session(
        session_id,
        request_id,
        selected_alg,
        key_material,
        expires_at,
        source_of_key
    )

    return CreateKeyResponse(
        session_id=session_id,
        request_id=request_id,
        requested_algorithm=req.algorithm,
        selected_algorithm=selected_alg,
        key_material=key_material,
        expires_at=expires_at,  # int
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        source_of_key=source_of_key
    )


@app.get("/kms/get_key/{session_id}", response_model=KeyResponse)
async def get_key_by_session(session_id: str):
    sess = await get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")

    return KeyResponse(
        session_id=sess["session_id"],
        request_id=sess.get("request_id", ""),
        algorithm=sess["algorithm"],
        key_material=sess["key_material"],
        expires_at=_to_unix_ts(sess["expires_at"])
    )


@app.get("/kms/get_key", response_model=KeyResponse)
async def get_key_by_request(request_id: str):
    sess = await get_session_by_request(request_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")

    return KeyResponse(
        session_id=sess["session_id"],
        request_id=sess.get("request_id", request_id),
        algorithm=sess["algorithm"],
        key_material=sess["key_material"],
        expires_at=_to_unix_ts(sess["expires_at"])
    )


@app.get("/kms/session/{session_id}", response_model=KeyResponse)
async def get_key(session_id: str):
    sess = await get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return KeyResponse(**sess)


@app.delete("/kms/session/{session_id}")
async def delete_key(session_id: str):
    from storage import delete_session
    success = await delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@app.get("/health")
def health_check():
    from key_manager import OQS_AVAILABLE, PQC_AVAILABLE
    import os

    status = {
        "status": "healthy",
        "components": {
            "liboqs": OQS_AVAILABLE,
            "pqcrypto": PQC_AVAILABLE,
            "qkd_gateway": os.getenv("QKD_AVAILABLE", "false").lower() == "true",
            "storage": True  # TODO: verify Redis/DB connectivity
        }
    }
    return status


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)