import base64, os
from datetime import datetime, timedelta
from crypto_utils import derive_key
import oqs  # liboqs para PQC

ALG_MAP = {
    "AES256_GCM": 32,
    "AES128_GCM": 16,
    "Kyber1024": 32,
    "Dilithium3": 48,
    "Falcon512": 32,
    "QKD_BB84": 32
}

def generate_key(algorithm: str):
    if algorithm.startswith("Kyber"):
        with oqs.KeyEncapsulation(algorithm) as kem:
            pk = kem.generate_keypair()
            ct, shared_key = kem.encap_secret(pk)
            return base64.b64encode(shared_key).decode()
    # outros PQC (Dilithium/Falcon) podem ir aqui

    # fallback random key
    size = ALG_MAP.get(algorithm, 32)
    raw = os.urandom(size)
    return base64.b64encode(derive_key(raw)).decode()

def build_session(session_id: str, algorithm: str, ttl: int):
    key_material = generate_key(algorithm)
    expires = datetime.utcnow() + timedelta(seconds=ttl)
    return session_id, algorithm, key_material, expires