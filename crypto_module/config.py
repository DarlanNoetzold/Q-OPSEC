import os

# Endereço do KMS (para recuperar material e metadados da chave)
KMS_BASE_URL = os.getenv("KMS_BASE_URL", "http://localhost:8002")

# Endereço do Context API (para buscar payload usando request_id)
CONTEXT_API_BASE_URL = os.getenv("CONTEXT_API_BASE_URL", "http://localhost:8005")

# Tempo limite para requests HTTP
HTTP_TIMEOUT = float(os.getenv("CRYPTO_HTTP_TIMEOUT", "10.0"))
CONTEXT_TIMEOUT = float(os.getenv("CRYPTO_CONTEXT_TIMEOUT", "8.0"))

# Tamanho padrão do nonce
AESGCM_NONCE_SIZE = 12  # 96 bits
CHACHA20_NONCE_SIZE = 12

# HKDF
HKDF_SALT = os.getenv("HKDF_SALT", "").encode() if os.getenv("HKDF_SALT") else None
HKDF_INFO_ENCRYPT = b"oraculumprisec:crypto:encrypt"
HKDF_INFO_DECRYPT = b"oraculumprisec:crypto:decrypt"

# Host/port do serviço
HOST = os.getenv("CRYPTO_HOST", "0.0.0.0")
PORT = int(os.getenv("CRYPTO_PORT", "8004"))