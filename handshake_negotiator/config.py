import os

KMS_URL = os.getenv("KMS_URL", "http://192.168.18.18:8002/kms/create_key")
KDE_URL = os.getenv("KDE_URL", "http://192.168.18.18:8003/deliver")
CRYPTO_URL = os.getenv("CRYPTO_URL", "http://192.168.18.18:8004/encrypt/by-request-id")
VALIDATION_URL = os.getenv("VALIDATION_URL", "http://192.168.18.18:8005/validation/send")

KMS_TIMEOUT = float(os.getenv("HANDSHAKE_KMS_TIMEOUT", "15.0"))
KDE_TIMEOUT = float(os.getenv("HANDSHAKE_KDE_TIMEOUT", "30.0"))
CRYPTO_TIMEOUT = float(os.getenv("HANDSHAKE_CRYPTO_TIMEOUT", "15.0"))
VALIDATION_TIMEOUT = float(os.getenv("HANDSHAKE_VALIDATION_TIMEOUT", "15.0"))
KEY_TTL_SECONDS = int(os.getenv("HANDSHAKE_KEY_TTL_SECONDS", "300"))
