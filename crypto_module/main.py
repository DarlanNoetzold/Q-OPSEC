import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

from models import EncryptRequest, EncryptResponse, DecryptRequest, DecryptResponse, ErrorResponse
from crypto_engine import (
    fetch_key_context,
    fetch_message_from_interceptor,
    aead_encrypt,
    aead_decrypt,
    b64d,
    b64e,
)
from config import HOST, PORT


def _to_unix_ts(value) -> int:
    """Convert various timestamp formats to Unix timestamp."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            raise ValueError(f"Invalid expires_at format: {value}")
    if isinstance(value, datetime):
        return int(value.timestamp())
    return int(value)


# Enhanced FastAPI app with comprehensive Swagger documentation
app = FastAPI(
    title="OraculumPrisec Crypto Module",
    description="""
    ## 🔐 Encryption/Decryption Service

    This service provides secure encryption and decryption operations using session keys from KMS 
    and messages from the Interceptor API.

    ### Features
    * **AES-256-GCM** and **ChaCha20-Poly1305** encryption algorithms
    * Integration with KMS for key management
    * Integration with Interceptor API for message retrieval
    * Authenticated Encryption with Associated Data (AEAD)
    * Base64 encoding for safe transport

    ### Workflow
    1. Fetch session key from KMS using `session_id`
    2. Optionally fetch message from Interceptor using `request_id`
    3. Perform encryption/decryption with AEAD
    4. Return base64-encoded results

    ### Security Notes
    * All keys are fetched from KMS - never stored locally
    * AAD (Additional Authenticated Data) ensures integrity
    * Nonces are randomly generated for each encryption
    """,
    version="1.3.1",
    contact={
        "name": "OraculumPrisec Security Team",
        "email": "security@oraculum.example.com",
    },
    license_info={
        "name": "Proprietary",
    },
    openapi_tags=[
        {
            "name": "Encryption",
            "description": "Operations for encrypting plaintext data using AEAD algorithms",
        },
        {
            "name": "Decryption",
            "description": "Operations for decrypting ciphertext back to plaintext",
        },
        {
            "name": "Health",
            "description": "Service health and status endpoints",
        },
    ],
)


def _default_aad(session_id: str, request_id: str, algorithm: str) -> bytes:
    """Generate default Additional Authenticated Data (AAD) for AEAD."""
    s = f"session:{session_id}|request:{request_id}|alg:{algorithm}"
    return s.encode("utf-8")


@app.post(
    "/encrypt",
    response_model=EncryptResponse,
    responses={
        200: {
            "description": "Successful encryption",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_abc123xyz",
                        "algorithm": "AES256_GCM",
                        "nonce_b64": "MTIzNDU2Nzg5MDEy",
                        "ciphertext_b64": "ZW5jcnlwdGVkX2RhdGFfaGVyZQ==",
                        "expires_at": 1708012800
                    }
                }
            }
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request - Invalid input parameters",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error - Encryption failed",
        }
    },
    tags=["Encryption"],
    summary="Encrypt data with flexible input options",
    description="""
    Encrypts plaintext data using AEAD (Authenticated Encryption with Associated Data).

    **Input Options:**
    1. Provide `plaintext_b64` directly
    2. Provide `request_id` to fetch message from Interceptor (requires `fetch_from_interceptor=True`)

    **Required:**
    - `session_id`: Used to fetch encryption key from KMS

    **Optional:**
    - `request_id`: Used to fetch message from Interceptor
    - `plaintext_b64`: Base64-encoded plaintext (if not fetching from Interceptor)
    - `algorithm`: Encryption algorithm (default: AES256_GCM)
    - `aad_b64`: Custom Additional Authenticated Data (auto-generated if not provided)
    - `fetch_from_interceptor`: Whether to fetch message from Interceptor (default: True)
    """,
)
async def encrypt(req: EncryptRequest):
    """
    Encrypt plaintext data using session key from KMS.

    The encryption process:
    1. Fetches session key context from KMS
    2. Optionally fetches message from Interceptor
    3. Generates or uses provided AAD
    4. Performs AEAD encryption
    5. Returns nonce and ciphertext (both base64-encoded)
    """
    try:
        if not req.session_id:
            raise ValueError("session_id is required to fetch key from KMS")

        ctx = await fetch_key_context(session_id=req.session_id, request_id=None)
        session_id = ctx["session_id"]
        req_id = req.request_id or ctx.get("request_id") or ""

        if not req.plaintext_b64:
            fetch_flag = getattr(req, "fetch_from_interceptor", True)
            if not fetch_flag:
                raise ValueError("plaintext_b64 is required when fetch_from_interceptor=False")
            if not req_id:
                raise ValueError("request_id is required to fetch message from Interceptor")

            intercepted = await fetch_message_from_interceptor(req_id)
            msg = intercepted.get("message")
            if msg is None:
                raise ValueError("Interceptor response missing 'message' field")
            payload_b64 = b64e(msg.encode("utf-8"))
        else:
            payload_b64 = req.plaintext_b64

        plaintext = b64d(payload_b64) or b""

        aad = b64d(req.aad_b64) if req.aad_b64 else _default_aad(session_id, req_id, req.algorithm)

        nonce_b64, ciphertext_b64 = aead_encrypt(ctx, req.algorithm, plaintext, aad)

        return EncryptResponse(
            session_id=session_id,
            algorithm=req.algorithm,
            nonce_b64=nonce_b64,
            ciphertext_b64=ciphertext_b64,
            expires_at=_to_unix_ts(ctx["expires_at"]),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encrypt error: {e}")


class EncryptByRequestId(BaseModel):
    """Request model for simplified encryption using request_id."""

    request_id: str = Field(
        ...,
        description="Request ID used to fetch message from Interceptor API",
        example="req_xyz789abc",
        min_length=1,
    )
    session_id: str = Field(
        ...,
        description="Session ID used to fetch encryption key from KMS",
        example="sess_abc123xyz",
        min_length=1,
    )
    algorithm: str = Field(
        "AES256_GCM",
        description="Encryption algorithm to use",
        example="AES256_GCM",
        pattern="^(AES256_GCM|CHACHA20_POLY1305)$",
    )
    aad_b64: Optional[str] = Field(
        None,
        description="Base64-encoded Additional Authenticated Data (auto-generated if not provided)",
        example="c2Vzc2lvbjphYmMxMjN8cmVxdWVzdDp4eXo3ODk=",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_xyz789abc",
                "session_id": "sess_abc123xyz",
                "algorithm": "AES256_GCM",
                "aad_b64": None
            }
        }


@app.post(
    "/encrypt/by-request-id",
    response_model=EncryptResponse,
    responses={
        200: {
            "description": "Successful encryption",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_abc123xyz",
                        "algorithm": "AES256_GCM",
                        "nonce_b64": "MTIzNDU2Nzg5MDEy",
                        "ciphertext_b64": "ZW5jcnlwdGVkX2RhdGFfaGVyZQ==",
                        "expires_at": 1708012800
                    }
                }
            }
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request - Invalid request_id or session_id",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error - Encryption or API call failed",
        }
    },
    tags=["Encryption"],
    summary="Simplified encryption using request_id",
    description="""
    Simplified encryption endpoint that automatically fetches the message from Interceptor.

    **Workflow:**
    1. Fetches message from Interceptor using `request_id`
    2. Fetches encryption key from KMS using `session_id`
    3. Encrypts the message using specified algorithm
    4. Returns encrypted data with nonce

    **Use this endpoint when:**
    - You have a `request_id` from the Interceptor
    - You want a simpler API without managing plaintext directly
    - You need automatic message retrieval
    """,
)
async def encrypt_by_request_id(body: EncryptByRequestId):
    """
    Encrypt message fetched from Interceptor using request_id.

    This is a convenience endpoint that combines message fetching and encryption
    in a single operation.
    """
    try:
        intercepted = await fetch_message_from_interceptor(body.request_id)
        msg = intercepted.get("message")
        if msg is None:
            raise ValueError("Interceptor response missing 'message' field")

        ctx = await fetch_key_context(session_id=body.session_id, request_id=None)

        aad = b64d(body.aad_b64) if body.aad_b64 else _default_aad(body.session_id, body.request_id, body.algorithm)

        nonce_b64, ciphertext_b64 = aead_encrypt(
            ctx,
            body.algorithm,
            msg.encode("utf-8"),
            aad,
        )

        return EncryptResponse(
            session_id=ctx["session_id"],
            algorithm=body.algorithm,
            nonce_b64=nonce_b64,
            ciphertext_b64=ciphertext_b64,
            expires_at=_to_unix_ts(ctx["expires_at"]),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encrypt error: {e}")


@app.post(
    "/decrypt",
    response_model=DecryptResponse,
    responses={
        200: {
            "description": "Successful decryption",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_abc123xyz",
                        "algorithm": "AES256_GCM",
                        "plaintext_b64": "SGVsbG8sIFdvcmxkIQ=="
                    }
                }
            }
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request - Invalid input or authentication failed",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error - Decryption failed",
        }
    },
    tags=["Decryption"],
    summary="Decrypt ciphertext using session key",
    description="""
    Decrypts ciphertext using AEAD (Authenticated Encryption with Associated Data).

    **Required:**
    - `session_id`: Used to fetch decryption key from KMS
    - `nonce_b64`: Base64-encoded nonce used during encryption
    - `ciphertext_b64`: Base64-encoded ciphertext to decrypt
    - `algorithm`: Algorithm used for encryption

    **Optional:**
    - `request_id`: Request ID for AAD generation
    - `aad_b64`: Custom Additional Authenticated Data (must match encryption AAD)

    **Important:**
    - The AAD must match exactly what was used during encryption
    - The nonce must be the same one returned from encryption
    - Authentication will fail if any parameter is incorrect
    """,
)
async def decrypt(req: DecryptRequest):
    """
    Decrypt ciphertext using session key from KMS.

    The decryption process:
    1. Fetches session key context from KMS
    2. Generates or uses provided AAD (must match encryption AAD)
    3. Performs AEAD decryption with authentication
    4. Returns base64-encoded plaintext

    Note: Decryption will fail if the ciphertext has been tampered with or
    if the AAD doesn't match the one used during encryption.
    """
    try:
        if not req.session_id:
            raise ValueError("session_id is required to fetch key from KMS")

        ctx = await fetch_key_context(session_id=req.session_id, request_id=None)
        session_id = ctx["session_id"]
        req_id = req.request_id or ctx.get("request_id") or ""

        aad = b64d(req.aad_b64) if req.aad_b64 else _default_aad(session_id, req_id, req.algorithm)

        plaintext = aead_decrypt(ctx, req.algorithm, req.nonce_b64, req.ciphertext_b64, aad)
        return DecryptResponse(
            session_id=session_id,
            algorithm=req.algorithm,
            plaintext_b64=b64e(plaintext),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decrypt error: {e}")


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns the current health status of the crypto module service.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "module": "crypto",
                        "version": "1.3.1"
                    }
                }
            }
        }
    },
)
async def health():
    """
    Check service health status.

    Returns basic information about the service including:
    - Status: Current health status
    - Module: Service module name
    - Version: Current version number
    """
    return {"status": "healthy", "module": "crypto", "version": "1.3.1"}


@app.get(
    "/",
    tags=["Health"],
    summary="API documentation redirect",
    description="Root endpoint that provides links to API documentation.",
    include_in_schema=True,
)
async def root():
    """
    Root endpoint with links to documentation.

    Provides quick access to:
    - Swagger UI (interactive API documentation)
    - ReDoc (alternative documentation view)
    - OpenAPI JSON schema
    """
    return {
        "service": "OraculumPrisec Crypto Module",
        "version": "1.3.1",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "encrypt": "/encrypt",
            "encrypt_by_request_id": "/encrypt/by-request-id",
            "decrypt": "/decrypt",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🔐 OraculumPrisec Crypto Module v1.3.1")
    print("=" * 60)
    print(f"🚀 Server starting on http://{HOST}:{PORT}")
    print(f"📚 Swagger UI: http://{HOST}:{PORT}/docs")
    print(f"📖 ReDoc: http://{HOST}:{PORT}/redoc")
    print(f"📄 OpenAPI JSON: http://{HOST}:{PORT}/openapi.json")
    print("=" * 60)

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)