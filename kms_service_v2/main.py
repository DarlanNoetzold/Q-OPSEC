import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException

from models import CreateKeyRequest, CreateKeyResponse, KeyResponse
from key_manager import build_session, get_supported_algorithms, get_algorithm_info
from storage import session_store
from hardware_profiler import get_hardware_summary, get_hardware_profile, run_algorithm_benchmarks
from crypto.pqc import OQS_AVAILABLE, PQC_AVAILABLE
from crypto.quantum import is_qkd_enabled
from quantum_gateway.netsquid_adapter import get_netsquid_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("qopsec.main")

app = FastAPI(
    title="Q-OPSEC Key Management Service",
    version="3.0.0",
    description="Adaptive cryptographic key management supporting Classical, Post-Quantum (PQC), and Quantum Key Distribution (QKD) algorithms.",
    openapi_tags=[
        {"name": "Algorithms", "description": "Algorithm discovery and metadata"},
        {"name": "Sessions", "description": "Key session creation and retrieval"},
        {"name": "Hardware", "description": "Hardware profiling and benchmarks"},
        {"name": "Health", "description": "Service health and component status"},
    ],
)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Q-OPSEC KMS",
        "version": "3.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "supported_algorithms": "/kms/supported_algorithms",
            "algorithm_info": "/kms/algorithm_info/{algorithm}",
            "create_key": "/kms/create_key",
            "get_key_by_session": "/kms/get_key/{session_id}",
            "get_key_by_request": "/kms/get_key?request_id=...",
            "hardware_profile": "/kms/hardware_profile",
            "health": "/health",
        },
    }


@app.get("/kms/supported_algorithms", tags=["Algorithms"])
def supported_algorithms():
    return get_supported_algorithms()


@app.get("/kms/algorithm_info/{algorithm}", tags=["Algorithms"])
def algorithm_info(algorithm: str):
    info = get_algorithm_info(algorithm)
    if not info:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm}' not found")
    return info


@app.post("/kms/create_key", response_model=CreateKeyResponse, tags=["Sessions"])
def create_key(req: CreateKeyRequest):
    try:
        (
            session_id,
            request_id,
            selected_algorithm,
            key_material,
            expires_at,
            fallback_applied,
            fallback_reason,
            source,
            qkd_metadata,
        ) = build_session(req.session_id, req.request_id, req.algorithm, req.ttl_seconds)

        session_store.save(
            session_id=session_id,
            request_id=request_id,
            algorithm=selected_algorithm,
            key_material=key_material,
            expires_at=expires_at,
            source=source,
        )

        return CreateKeyResponse(
            session_id=session_id,
            request_id=request_id,
            requested_algorithm=req.algorithm,
            selected_algorithm=selected_algorithm,
            key_material=key_material,
            expires_at=expires_at,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            source_of_key=source,
            qkd_metadata=qkd_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error creating key session")
        raise HTTPException(status_code=500, detail="Internal error during key generation")


@app.get("/kms/get_key/{key_id}", response_model=KeyResponse, tags=["Sessions"])
def get_key_by_session(key_id: str):
    session = session_store.get_by_session_id(key_id)
    if not session:
        session = session_store.get_by_request_id(key_id)
    if not session:
        raise HTTPException(status_code=404, detail="Key session not found or expired")

    return KeyResponse(
        session_id=session["session_id"],
        request_id=session.get("request_id", ""),
        algorithm=session["algorithm"],
        key_material=session["key_material"],
        expires_at=session["expires_at"],
    )


@app.get("/kms/get_key", response_model=KeyResponse, tags=["Sessions"])
def get_key_by_request(request_id: str):
    session = session_store.get_by_request_id(request_id)
    if not session:
        raise HTTPException(status_code=404, detail="Key session not found or expired")

    return KeyResponse(
        session_id=session["session_id"],
        request_id=session.get("request_id", request_id),
        algorithm=session["algorithm"],
        key_material=session["key_material"],
        expires_at=session["expires_at"],
    )


@app.delete("/kms/session/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    success = session_store.delete(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@app.get("/kms/hardware_profile", tags=["Hardware"])
def hardware_profile():
    return get_hardware_summary()


@app.post("/kms/benchmark", tags=["Hardware"])
def run_benchmarks(algorithms: list[str] = None):
    results = run_algorithm_benchmarks(algorithms)
    return {
        algo: {
            "keygen_time_ms": round(bench.keygen_time_ms, 3),
            "throughput_ops_per_sec": round(bench.throughput_ops_per_sec, 2),
            "availability": bench.availability.value,
        }
        for algo, bench in results.items()
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "components": {
            "liboqs": OQS_AVAILABLE,
            "pqcrypto": PQC_AVAILABLE,
            "qkd_gateway": is_qkd_enabled(),
            "netsquid": get_netsquid_status(),
            "storage": session_store.stats(),
            "hardware": get_hardware_summary(),
        },
    }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Q-OPSEC KMS v3.0.0")
    logger.info("Swagger UI: http://0.0.0.0:8002/docs")
    logger.info("ReDoc: http://0.0.0.0:8002/redoc")
    logger.info("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
