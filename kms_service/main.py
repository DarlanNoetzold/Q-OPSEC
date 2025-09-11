from fastapi import FastAPI, HTTPException, Response
import uvicorn
from models import CreateKeyRequest, CreateKeyResponse, KeyResponse
from key_manager import build_session, get_supported_algorithms, get_algorithm_info
from storage import save_session, get_session

app = FastAPI(title="KMS Service", version="2.0.0")


@app.get("/kms/supported_algorithms")
def supported_algorithms():
    """List all algorithms supported by the KMS"""
    return get_supported_algorithms()


@app.get("/kms/algorithm_info/{algorithm}")
def algorithm_info(algorithm: str):
    """Return detailed information about a specific algorithm"""
    return get_algorithm_info(algorithm)


@app.post("/kms/create_key", response_model=CreateKeyResponse)
async def create_key(req: CreateKeyRequest, response: Response):
    """
    Create a new session key using the specified algorithm.

    - If strict=True and a fallback occurs, returns 409
    - Otherwise, applies fallback and informs in the response
    """
    try:
        sid, selected_alg, key_material, expires_at, fb_applied, fb_reason, src = build_session(
            req.session_id, req.algorithm, req.ttl_seconds
        )

        # Strict mode → error if fallback was applied
        if req.strict and fb_applied:
            raise HTTPException(
                status_code=409,
                detail=f"Requested '{req.algorithm}' but used '{selected_alg}' due to {fb_reason}"
            )

        # Add observability headers
        response.headers["X-KMS-Requested-Algorithm"] = req.algorithm
        response.headers["X-KMS-Selected-Algorithm"] = selected_alg
        if fb_applied:
            response.headers["X-KMS-Fallback"] = fb_reason or "UNKNOWN"

        # Save session in storage
        await save_session({
            "session_id": sid,
            "algorithm": selected_alg,
            "key_material": key_material,
            "expires_at": expires_at,
            "source": src
        })

        return CreateKeyResponse(
            session_id=sid,
            requested_algorithm=req.algorithm,
            selected_algorithm=selected_alg,
            key_material=key_material,
            expires_at=expires_at,
            fallback_applied=fb_applied,
            fallback_reason=fb_reason,
            source_of_key=src,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kms/session/{session_id}", response_model=KeyResponse)
async def get_key(session_id: str):
    """Retrieve an existing session key by ID"""
    sess = await get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return KeyResponse(**sess)


@app.delete("/kms/session/{session_id}")
async def delete_key(session_id: str):
    """Remove a session key (revocation)"""
    from storage import delete_session
    success = await delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@app.get("/health")
def health_check():
    """KMS health check - verifies component status"""
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