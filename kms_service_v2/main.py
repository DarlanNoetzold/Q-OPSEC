"""
Q-OPSEC Key Management Service (KMS) v2.0.0

Endpoints:
  POST /kms/create_key          — Create a key session
  GET  /kms/get_key/{id}        — Retrieve key by session_id or request_id
  GET  /kms/get_key?request_id= — Retrieve key by request_id (query param)
  GET  /kms/supported_algorithms — List supported algorithms
  GET  /kms/algorithm_info/{alg} — Algorithm details
  GET  /health                   — Health check
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Optional

from models import CreateKeyRequest, CreateKeyResponse, KeyResponse
from key_manager import build_session, get_supported_algorithms, get_algorithm_info
from storage import save_session, get_session, get_session_by_request

app = FastAPI(
    title="Q-OPSEC KMS Service",
    version="2.0.0",
    description="""
## 🔐 Key Management Service (KMS)

Serviço para criação e consulta de **sessões de chave** para criptografia adaptativa.

### Algoritmos suportados
- **Clássico:** AES256_GCM, AES128_GCM, ChaCha20_Poly1305, RSA, ECC
- **Pós-Quântico (PQC):** Kyber512, Kyber768, Kyber1024
- **Quântico (QKD):** QKD_BB84, QKD_E91, QKD_CV, QKD_MDI

### Documentação
- Swagger UI: `/docs`
- ReDoc: `/redoc`
""",
    openapi_tags=[
        {"name": "Algorithms", "description": "Descoberta de algoritmos e metadados"},
        {"name": "Sessions", "description": "Criação/consulta/remoção de sessões de chave"},
        {"name": "Health", "description": "Endpoints de saúde"},
    ],
)


@app.get("/", tags=["Health"], summary="Root — links úteis")
def root():
    return {
        "service": "Q-OPSEC KMS Service",
        "version": "2.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "supported_algorithms": "/kms/supported_algorithms",
            "algorithm_info": "/kms/algorithm_info/{algorithm}",
            "create_key": "/kms/create_key",
            "get_key": "/kms/get_key/{session_id}",
            "health": "/health",
        },
    }


@app.get("/kms/supported_algorithms", tags=["Algorithms"],
         summary="Lista algoritmos suportados")
def supported_algorithms():
    return get_supported_algorithms()


@app.get("/kms/algorithm_info/{algorithm}", tags=["Algorithms"],
         summary="Detalhes de um algoritmo")
def algorithm_info(algorithm: str):
    info = get_algorithm_info(algorithm)
    if not info:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return info


@app.post("/kms/create_key", response_model=CreateKeyResponse, tags=["Sessions"],
          summary="Cria uma sessão de chave")
async def create_key(req: CreateKeyRequest):
    try:
        (
            session_id,
            request_id,
            selected_alg,
            key_material,
            expires_at,
            fallback_applied,
            fallback_reason,
            source_of_key,
        ) = build_session(req.session_id, req.request_id, req.algorithm, req.ttl_seconds)

        await save_session(
            session_id, request_id, selected_alg, key_material, expires_at, source_of_key,
        )

        return CreateKeyResponse(
            session_id=session_id,
            request_id=request_id,
            requested_algorithm=req.algorithm,
            selected_algorithm=selected_alg,
            key_material=key_material,
            expires_at=expires_at,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            source_of_key=source_of_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/kms/get_key/{key_id}", response_model=KeyResponse, tags=["Sessions"],
         summary="Consulta chave por session_id ou request_id")
async def get_key_by_id(key_id: str):
    """Busca sessão por session_id ou request_id (path param)."""
    sess = await get_session(key_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")

    return KeyResponse(
        session_id=sess["session_id"],
        request_id=sess.get("request_id", ""),
        algorithm=sess["algorithm"],
        key_material=sess["key_material"],
        expires_at=int(sess["expires_at"]),
    )


@app.get("/kms/get_key", response_model=KeyResponse, tags=["Sessions"],
         summary="Consulta chave por request_id (query param)")
async def get_key_by_request(request_id: str):
    sess = await get_session_by_request(request_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")

    return KeyResponse(
        session_id=sess["session_id"],
        request_id=sess.get("request_id", request_id),
        algorithm=sess["algorithm"],
        key_material=sess["key_material"],
        expires_at=int(sess["expires_at"]),
    )


@app.delete("/kms/session/{session_id}", tags=["Sessions"],
            summary="Remove uma sessão")
async def delete_key(session_id: str):
    from storage import delete_session
    success = await delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@app.get("/health", tags=["Health"], summary="Health check")
def health_check():
    from key_manager import OQS_AVAILABLE, PQC_AVAILABLE
    import os
    return {
        "status": "healthy",
        "components": {
            "liboqs": OQS_AVAILABLE,
            "pqcrypto": PQC_AVAILABLE,
            "qkd_gateway": os.getenv("QKD_AVAILABLE", "true").lower() == "true",
            "storage": True,
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🔐 Q-OPSEC KMS Service v2.0.0")
    print("=" * 60)
    print("📚 Swagger UI: http://0.0.0.0:8002/docs")
    print("📖 ReDoc:      http://0.0.0.0:8002/redoc")
    print("=" * 60)

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
