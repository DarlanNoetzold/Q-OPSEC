from prometheus_fastapi_instrumentator import Instrumentator
import httpx
import uvicorn
import logging
from uuid import uuid4
from fastapi import FastAPI, HTTPException

from urllib.parse import urlsplit, urlunsplit

from models import NegotiationRequest, NegotiationResponse
from negotiator import negotiate_algorithms

app = FastAPI(
    title="Handshake Negotiator",
    version="2.3.0",
    description="""
## 🤝 Handshake Negotiator Service

Serviço de **negociação de algoritmos criptográficos** e orquestração do fluxo completo de:
1. **Negociação** de algoritmo entre cliente e servidor
2. **Criação de chave** no KMS
3. **Entrega de chave** via KDE
4. **Criptografia** da mensagem via Crypto Module
5. **Validação** e envio para o receptor

### Fluxo Completo
```
Cliente → Handshake → KMS → KDE → Crypto → Validation → Receptor
```

### Integrações
- **KMS** (porta 8002): Criação de sessões de chave
- **KDE** (porta 8003): Entrega de chaves
- **Crypto** (porta 8004): Criptografia de mensagens
- **Validation** (porta 8005): Validação e envio final

### Documentação
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
""",
    contact={
        "name": "Q-OPSEC Team",
        "email": "security@qopsec.example.com",
    },
    openapi_tags=[
        {
            "name": "Negotiation",
            "description": "Endpoints de negociação e orquestração do handshake completo",
        },
        {
            "name": "Health",
            "description": "Endpoints de saúde e informações do serviço",
        },
    ],
)

KMS_URL = "http://192.168.18.18:8002/kms/create_key"
KDE_URL = "http://192.168.18.18:8003/deliver"
CRYPTO_URL = "http://192.168.18.18:8004/encrypt/by-request-id"
VALIDATION_URL = "http://192.168.18.18:8005/validation/send"


def normalize_destination(dest: str) -> str:
    """
    Normaliza a URL de destino, adicionando /receiver se necessário.

    Args:
        dest: URL de destino original

    Returns:
        URL normalizada com path /receiver se aplicável
    """
    try:
        parts = urlsplit(dest)
        if parts.scheme not in ("http", "https"):
            return dest
        path = parts.path or ""
        if path.strip() in ("", "/"):
            path = "/receiver"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    except Exception:
        return dest


@app.get(
    "/",
    tags=["Health"],
    summary="Informações do serviço e links de documentação",
    description="Endpoint raiz com informações sobre o serviço e links úteis.",
)
def root():
    """
    Retorna informações básicas do serviço e links para documentação.
    """
    return {
        "service": "Handshake Negotiator",
        "version": "2.3.0",
        "description": "Serviço de negociação de algoritmos e orquestração de handshake criptográfico",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "handshake": "/handshake",
            "health": "/health",
        },
        "integrations": {
            "kms": KMS_URL,
            "kde": KDE_URL,
            "crypto": CRYPTO_URL,
            "validation": VALIDATION_URL,
        },
    }


@app.post(
    "/handshake",
    response_model=NegotiationResponse,
    tags=["Negotiation"],
    summary="Executa handshake completo com negociação de algoritmo",
    description="""
Orquestra o fluxo completo de handshake criptográfico:

### Etapas do Processo

1. **Negociação de Algoritmo**
   - Recebe lista de algoritmos propostos pelo cliente
   - Negocia o melhor algoritmo disponível
   - Aplica fallback se necessário

2. **Criação de Chave (KMS)**
   - Cria sessão de chave no KMS
   - Obtém `session_id`, `key_material`, `expires_at`
   - Registra fonte da chave (QKD, PQC, Classical)

3. **Entrega de Chave (KDE)**
   - Envia chave para o destino via KDE
   - Normaliza URL de destino (adiciona `/receiver` se necessário)
   - Método de entrega: API

4. **Criptografia (Crypto Module)**
   - Busca mensagem do Interceptor usando `request_id`
   - Criptografa usando a chave da sessão
   - Retorna `nonce_b64` e `ciphertext_b64`

5. **Validação e Envio Final**
   - Envia dados criptografados para o serviço de Validação
   - Validação encaminha para o receptor final
   - Inclui todos os metadados necessários para decriptação

### Parâmetros de Entrada

- `proposed`: Lista de algoritmos propostos (ex: `["KYBER1024", "AES256_GCM"]`)
- `destination`: URL do receptor (ex: `http://receiver.example.com`)
- `request_id`: ID da requisição (opcional, gerado automaticamente se não fornecido)
- `source_id`: ID da origem (opcional)

### Resposta

Retorna objeto completo com:
- IDs de sessão e requisição
- Algoritmo selecionado e fallback (se aplicado)
- Material de chave e expiração
- Dados de criptografia (nonce, ciphertext)
- Status de entrega (KDE + Validation)
""",
    responses={
        200: {
            "description": "Handshake completado com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "req_abc123xyz",
                        "session_id": "sess_xyz789abc",
                        "requested_algorithm": "KYBER1024",
                        "selected_algorithm": "AES256_GCM",
                        "key_material": "***",
                        "expires_at": 1708012800,
                        "fallback_applied": True,
                        "fallback_reason": "KYBER1024 not available",
                        "source_of_key": "classical",
                        "message": "Negotiation completed with fallback: KYBER1024 -> AES256_GCM",
                        "delivery_status": "{'kde': {...}, 'validation': {...}}",
                        "crypto_nonce_b64": "MTIzNDU2Nzg5MDEy",
                        "crypto_ciphertext_b64": "ZW5jcnlwdGVkX2RhdGE=",
                        "crypto_algorithm": "AES256_GCM",
                        "crypto_expires_at": 1708012800,
                    }
                }
            },
        },
        400: {
            "description": "Requisição inválida - parâmetros faltando ou inválidos",
        },
        500: {
            "description": "Erro interno - falha em algum serviço integrado (KMS, KDE, Crypto, Validation)",
        },
    },
)
async def handshake(req: NegotiationRequest):
    req_id = req.request_id or "req-unknown"
    print(f"[{req_id}] Handshake: Processing request from {req.source}")
    """
    Executa o handshake completo de negociação criptográfica.

    Este endpoint orquestra todo o fluxo de:
    - Negociação de algoritmo
    - Criação de chave no KMS
    - Entrega de chave via KDE
    - Criptografia da mensagem
    - Validação e envio para o receptor

    Args:
        req: Objeto NegotiationRequest contendo algoritmos propostos e destino

    Returns:
        NegotiationResponse com todos os dados do handshake completado

    Raises:
        HTTPException: Se algum serviço integrado falhar
    """
    request_id = req.request_id or f"req_{uuid4()}"
    logging.info(f"[{request_id}] Handshake started: proposed={req.proposed}")

    requested_alg = req.proposed[0] if req.proposed else "UNKNOWN"
    chosen_alg, session_id, _, _ = negotiate_algorithms(req)


    kms_payload = {
        "session_id": session_id,
        "request_id": request_id,
        "algorithm": chosen_alg,
        "ttl_seconds": 300
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        kms_resp = await client.post(KMS_URL, json=kms_payload)
    if kms_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Erro ao criar chave no KMS: {kms_resp.text}")
    try:
        key_data = kms_resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resposta inválida do KMS: {e}")

    actual_selected = key_data.get("selected_algorithm", chosen_alg)
    actual_fallback = key_data.get("fallback_applied", False)
    actual_reason = key_data.get("fallback_reason")
    actual_source = key_data.get("source_of_key", "unknown")

    message = (
        f"Negotiation completed with fallback: {requested_alg} -> {actual_selected}"
        if actual_fallback else
        "Negotiation completed successfully"
    )

    raw_dest = req.destination
    if raw_dest and "10.0.0.5" in raw_dest:
        raw_dest = raw_dest.replace("10.0.0.5", "192.168.18.18")
    normalized_dest = normalize_destination(raw_dest)
    delivery_payload = {
        "session_id": key_data["session_id"],
        "request_id": request_id,
        "destination": normalized_dest,
        "delivery_method": "API",
        "key_material": key_data["key_material"],
        "algorithm": actual_selected,
        "expires_at": key_data["expires_at"]
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        kde_resp = await client.post(KDE_URL, json=delivery_payload)
    if kde_resp.status_code != 200:
        kde_data = {"error": f"HTTP {kde_resp.status_code}", "body": kde_resp.text}
    else:
        try:
            kde_data = kde_resp.json()
        except Exception:
            kde_data = {"raw": kde_resp.text}

    crypto_payload = {
        "request_id": request_id,
        "session_id": key_data["session_id"],
        "algorithm": actual_selected
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        crypto_resp = await client.post(CRYPTO_URL, json=crypto_payload)

    crypto_nonce_b64 = None
    crypto_ciphertext_b64 = None
    crypto_algorithm = None
    crypto_expires_at = None

    if crypto_resp.status_code == 200:
        try:
            c = crypto_resp.json()
            crypto_nonce_b64 = c.get("nonce_b64")
            crypto_ciphertext_b64 = c.get("ciphertext_b64")
            crypto_algorithm = c.get("algorithm")
            crypto_expires_at = c.get("expires_at")
        except Exception:
            pass

    validation_data: dict | str
    if crypto_nonce_b64 and crypto_ciphertext_b64:
        validation_payload = {
            "requestId": request_id,
            "sessionId": key_data["session_id"],
            "selectedAlgorithm": actual_selected,
            "cryptoNonceB64": crypto_nonce_b64,
            "cryptoCiphertextB64": crypto_ciphertext_b64,
            "cryptoAlgorithm": crypto_algorithm or actual_selected,
            "cryptoExpiresAt": crypto_expires_at,
            "sourceId": getattr(req, "source_id", None),
            "originUrl": normalized_dest
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            v_resp = await client.post(VALIDATION_URL, json=validation_payload)
        if v_resp.status_code != 200:
            validation_data = {"error": f"HTTP {v_resp.status_code}", "body": v_resp.text}
        else:
            try:
                validation_data = v_resp.json()
            except Exception:
                validation_data = {"raw": v_resp.text}
    else:
        validation_data = {"skip": "no crypto output available"}

    return NegotiationResponse(
        request_id=request_id,
        session_id=key_data["session_id"],
        requested_algorithm=requested_alg,
        selected_algorithm=actual_selected,
        key_material=key_data["key_material"],
        expires_at=key_data["expires_at"],
        fallback_applied=actual_fallback,
        fallback_reason=actual_reason,
        source_of_key=actual_source,
        message=message,
        delivery_status=str({
            "kde": kde_data,
            "validation": validation_data
        }),
        crypto_nonce_b64=crypto_nonce_b64,
        crypto_ciphertext_b64=crypto_ciphertext_b64,
        crypto_algorithm=crypto_algorithm,
        crypto_expires_at=crypto_expires_at,
    )


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check do serviço",
    description="Verifica o status de saúde do Handshake Negotiator e conectividade com serviços integrados.",
    responses={
        200: {
            "description": "Serviço saudável",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "handshake-negotiator",
                        "version": "2.3.0",
                        "integrations": {
                            "kms": "http://localhost:8002/kms/create_key",
                            "kde": "http://localhost:8003/deliver",
                            "crypto": "http://localhost:8004/encrypt/by-request-id",
                            "validation": "http://localhost:8005/validation/send",
                        },
                    }
                }
            },
        }
    },
)
async def health_check():
    """
    Verifica o status de saúde do serviço.

    Retorna informações sobre:
    - Status geral do serviço
    - Versão atual
    - URLs dos serviços integrados

    Note: Este endpoint não verifica conectividade real com os serviços integrados.
    Para verificação completa, use o endpoint /handshake com dados de teste.
    """
    return {
        "status": "healthy",
        "service": "handshake-negotiator",
        "version": "2.3.0",
        "integrations": {
            "kms": KMS_URL,
            "kde": KDE_URL,
            "crypto": CRYPTO_URL,
            "validation": VALIDATION_URL,
        },
    }


if __name__ == "__main__":
    print("=" * 70)
    print("🤝 Handshake Negotiator v2.3.0")
    print("=" * 70)
    print("🚀 Server starting on http://0.0.0.0:8001")
    print("📚 Swagger UI:    http://0.0.0.0:8001/docs")
    print("📖 ReDoc:         http://0.0.0.0:8001/redoc")
    print("📄 OpenAPI JSON:  http://0.0.0.0:8001/openapi.json")
    print("=" * 70)
    print("🔗 Integrated Services:")
    print(f"   • KMS:        {KMS_URL}")
    print(f"   • KDE:        {KDE_URL}")
    print(f"   • Crypto:     {CRYPTO_URL}")
    print(f"   • Validation: {VALIDATION_URL}")
    print("=" * 70)

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)


try:
    Instrumentator().instrument(app).expose(app)
except Exception as e:
    print(f'Monitoring Error: {e}')
