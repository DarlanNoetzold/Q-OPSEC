import uvicorn
from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import Any, Optional

from models import CreateKeyRequest, CreateKeyResponse, KeyResponse
from key_manager import build_session, get_supported_algorithms, get_algorithm_info
from storage import save_session, get_session, get_session_by_request

app = FastAPI(
    title="KMS Service",
    version="2.0.0",
    description="""
## 🔐 Key Management Service (KMS)

Serviço para criação e consulta de **sessões de chave** para criptografia/AEAD.

### Principais capacidades
- Criar uma sessão de chave (`/kms/create_key`)
- Consultar sessão por `session_id` ou `request_id`
- Listar algoritmos suportados e obter detalhes de algoritmos

### Documentação
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
""",
    openapi_tags=[
        {"name": "Algorithms", "description": "Descoberta de algoritmos e metadados"},
        {"name": "Sessions", "description": "Criação/consulta/remoção de sessões de chave"},
        {"name": "Health", "description": "Endpoints de saúde e verificação de dependências"},
    ],
)


def _to_unix_ts(value) -> int:
    """Convert various timestamp formats to Unix timestamp."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    return int(value)


@app.get(
    "/",
    tags=["Health"],
    summary="Links rápidos para documentação e endpoints",
    description="Endpoint raiz com links úteis para Swagger, ReDoc e OpenAPI.",
)
def root():
    return {
        "service": "KMS Service",
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
            "get_key_by_session": "/kms/get_key/{session_id}",
            "get_key_by_request": "/kms/get_key?request_id=...",
            "session_get": "/kms/session/{session_id}",
            "session_delete": "/kms/session/{session_id}",
            "health": "/health",
        },
    }


@app.get(
    "/kms/supported_algorithms",
    tags=["Algorithms"],
    summary="Lista algoritmos suportados",
    description="Retorna a lista (ou estrutura) de algoritmos suportados pelo serviço.",
    responses={
        200: {
            "description": "Lista/estrutura de algoritmos suportados",
        }
    },
)
def supported_algorithms():
    return get_supported_algorithms()


@app.get(
    "/kms/algorithm_info/{algorithm}",
    tags=["Algorithms"],
    summary="Detalhes de um algoritmo",
    description="Retorna metadados/detalhes do algoritmo informado (ex.: parâmetros, disponibilidade, etc.).",
    responses={
        200: {"description": "Informações do algoritmo"},
        404: {"description": "Algoritmo não encontrado"},
        400: {"description": "Parâmetro inválido"},
    },
)
def algorithm_info(algorithm: str):
    # Se get_algorithm_info já lança erro, ok.
    # Se não, você pode ajustar aqui para retornar 404 quando None.
    info = get_algorithm_info(algorithm)
    if not info:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return info


@app.post(
    "/kms/create_key",
    response_model=CreateKeyResponse,
    tags=["Sessions"],
    summary="Cria uma sessão de chave",
    description="""
Cria (ou negocia) uma sessão contendo:
- `session_id`
- `request_id`
- `selected_algorithm`
- `key_material`
- `expires_at`

Pode aplicar fallback de algoritmo, dependendo da disponibilidade.
""",
    responses={
        200: {
            "description": "Sessão criada com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_abc123xyz",
                        "request_id": "req_xyz789abc",
                        "requested_algorithm": "AES256_GCM",
                        "selected_algorithm": "AES256_GCM",
                        "key_material": "base64-or-hex-material",
                        "expires_at": 1708012800,
                        "fallback_applied": False,
                        "fallback_reason": None,
                        "source_of_key": "qkd|pqc|classical|mock",
                    }
                }
            },
        },
        400: {"description": "Requisição inválida"},
        500: {"description": "Erro interno ao criar sessão"},
    },
)
async def create_key(req: CreateKeyRequest):
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
        session_id,
        request_id,
        selected_alg,
        key_material,
        expires_at,
        source_of_key,
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
        source_of_key=source_of_key,
    )


@app.get(
    "/kms/get_key/{session_id}",
    response_model=KeyResponse,
    tags=["Sessions"],
    summary="Consulta chave por session_id",
    description="Busca a sessão (se existir e não estiver expirada) usando `session_id`.",
    responses={
        200: {"description": "Sessão encontrada"},
        404: {"description": "Não encontrada ou expirada"},
    },
)
async def get_key_by_session(session_id: str):
    sess = await get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")

    return KeyResponse(
        session_id=sess["session_id"],
        request_id=sess.get("request_id", ""),
        algorithm=sess["algorithm"],
        key_material=sess["key_material"],
        expires_at=_to_unix_ts(sess["expires_at"]),
    )


@app.get(
    "/kms/get_key",
    response_model=KeyResponse,
    tags=["Sessions"],
    summary="Consulta chave por request_id",
    description="Busca a sessão usando `request_id` como query parameter.",
    responses={
        200: {"description": "Sessão encontrada"},
        404: {"description": "Não encontrada ou expirada"},
        422: {"description": "Parâmetros inválidos (FastAPI validation)"},
    },
)
async def get_key_by_request(request_id: str):
    sess = await get_session_by_request(request_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Not found or expired")

    return KeyResponse(
        session_id=sess["session_id"],
        request_id=sess.get("request_id", request_id),
        algorithm=sess["algorithm"],
        key_material=sess["key_material"],
        expires_at=_to_unix_ts(sess["expires_at"]),
    )


@app.get(
    "/kms/session/{session_id}",
    response_model=KeyResponse,
    tags=["Sessions"],
    summary="Consulta sessão (raw) por session_id",
    description="Retorna a sessão inteira (mesmos campos do armazenamento), se existir.",
    responses={
        200: {"description": "Sessão encontrada"},
        404: {"description": "Sessão não encontrada ou expirada"},
    },
)
async def get_key(session_id: str):
    sess = await get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return KeyResponse(**sess)


@app.delete(
    "/kms/session/{session_id}",
    tags=["Sessions"],
    summary="Remove uma sessão",
    description="Apaga uma sessão existente por `session_id`.",
    responses={
        200: {
            "description": "Sessão deletada",
            "content": {"application/json": {"example": {"message": "Session deleted successfully"}}},
        },
        404: {"description": "Sessão não encontrada"},
    },
)
async def delete_key(session_id: str):
    from storage import delete_session

    success = await delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Retorna estado de saúde do serviço e disponibilidade de componentes (liboqs, pqcrypto, etc.).",
    responses={
        200: {
            "description": "Status do serviço",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "components": {
                            "liboqs": True,
                            "pqcrypto": True,
                            "qkd_gateway": False,
                            "storage": True,
                        },
                    }
                }
            },
        }
    },
)
def health_check():
    from key_manager import OQS_AVAILABLE, PQC_AVAILABLE
    import os

    status = {
        "status": "healthy",
        "components": {
            "liboqs": OQS_AVAILABLE,
            "pqcrypto": PQC_AVAILABLE,
            "qkd_gateway": os.getenv("QKD_AVAILABLE", "false").lower() == "true",
            "storage": True,  # TODO: verify Redis/DB connectivity
        },
    }
    return status


if __name__ == "__main__":
    print("=" * 60)
    print("🔐 KMS Service v2.0.0")
    print("=" * 60)
    print("📚 Swagger UI: http://0.0.0.0:8002/docs")
    print("📖 ReDoc:      http://0.0.0.0:8002/redoc")
    print("📄 OpenAPI:    http://0.0.0.0:8002/openapi.json")
    print("="* 60)

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)