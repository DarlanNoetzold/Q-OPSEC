import base64
from datetime import datetime, timedelta
from crypto_utils import derive_key
from quantum_gateway.gateway import generate_key_from_gateway
from quantum_gateway.compatibility_layer import to_session_key
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
import oqs


def generate_key(algorithm: str):

    # 🔹 Primeiro tenta Quantum Gateway
    chosen_algo, key = generate_key_from_gateway(algorithm)
    if key:
        return chosen_algo, key

    # 🔹 Pós-Quânticos (Key Encapsulation Mechanisms)
    if algorithm in oqs.get_enabled_KEMs():
        with oqs.KeyEncapsulation(algorithm) as kem:
            public_key = kem.generate_keypair()
            ciphertext, session_key = kem.encap_secret(public_key)
            return algorithm, to_session_key(session_key)

    # 🔹 PQC Signatures – não KEM, mas podemos derivar entropy
    if algorithm in oqs.get_enabled_Sigs():
        with oqs.Signature(algorithm) as sig:
            public_key = sig.generate_keypair()
            return algorithm, to_session_key(derive_key(public_key, 32))

    # 🔹 Clássicos
    if algorithm == "AES256_GCM":
        key = AESGCM.generate_key(bit_length=256)
        return algorithm, base64.b64encode(key).decode()

    if algorithm == "AES128_GCM":
        key = AESGCM.generate_key(bit_length=128)
        return algorithm, base64.b64encode(key).decode()

    if algorithm == "ChaCha20_Poly1305":
        key = ChaCha20Poly1305.generate_key()
        return algorithm, base64.b64encode(key).decode()

    raise ValueError(f"Algoritmo {algorithm} não suportado.")


def build_session(session_id: str, algorithm: str, ttl: int = 300):
    alg, key_material = generate_key(algorithm)
    expires = datetime.utcnow() + timedelta(seconds=ttl)
    return session_id, alg, key_material, expires