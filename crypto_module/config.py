import os

KMS_BASE_URL = os.getenv("KMS_BASE_URL", "http://localhost:8002")

CONTEXT_API_BASE_URL = os.getenv("CONTEXT_API_BASE_URL", "http://localhost:8081")

HTTP_TIMEOUT = float(os.getenv("CRYPTO_HTTP_TIMEOUT", "10.0"))
CONTEXT_TIMEOUT = float(os.getenv("CRYPTO_CONTEXT_TIMEOUT", "8.0"))

AESGCM_NONCE_SIZE = 12
CHACHA20_NONCE_SIZE = 12

HKDF_SALT = os.getenv("HKDF_SALT", "").encode() if os.getenv("HKDF_SALT") else None
HKDF_INFO_ENCRYPT = b"oraculumprisec:crypto:encrypt"
HKDF_INFO_DECRYPT = b"oraculumprisec:crypto:decrypt"

HOST = os.getenv("CRYPTO_HOST", "0.0.0.0")
PORT = int(os.getenv("CRYPTO_PORT", "8004"))