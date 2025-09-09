import base64
from datetime import datetime, timedelta
from crypto_utils import derive_key
from quantum_gateway.gateway import generate_key_from_gateway
from quantum_gateway.compatibility_layer import to_session_key
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

# Tenta importar oqs
try:
    import oqs  # Open Quantum Safe
    OQS_AVAILABLE = True
except Exception:
    oqs = None
    OQS_AVAILABLE = False


def _kem_mechanisms():
    if not OQS_AVAILABLE:
        return []
    for fn in [
        "get_available_KEM_mechanisms",
        "get_enabled_KEM_mechanisms",
        "get_supported_KEM_mechanisms",
        "get_enabled_KEMs",  # APIs antigas
    ]:
        mech_getter = getattr(oqs, fn, None)
        if mech_getter:
            try:
                mechs = mech_getter()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                pass
    return []


def _sig_mechanisms():
    if not OQS_AVAILABLE:
        return []
    for fn in [
        "get_available_sig_mechanisms",
        "get_enabled_sig_mechanisms",
        "get_supported_sig_mechanisms",
        "get_enabled_Sigs",
    ]:
        mech_getter = getattr(oqs, fn, None)
        if mech_getter:
            try:
                mechs = mech_getter()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                pass
    return []


def generate_key(algorithm: str):
    # 1) QKD via Quantum Gateway
    chosen_algo, key_b64_or_bytes = generate_key_from_gateway(algorithm)
    if key_b64_or_bytes:
        return chosen_algo, key_b64_or_bytes

    # 2) PQC KEM
    kem_mechs = _kem_mechanisms()
    if kem_mechs and algorithm in kem_mechs:
        with oqs.KeyEncapsulation(algorithm) as kem:
            public_key = kem.generate_keypair()
            ciphertext, session_key = kem.encap_secret(public_key)
            return algorithm, base64.b64encode(session_key).decode()

    # 3) PQC Signatures (deriva material)
    sig_mechs = _sig_mechanisms()
    if sig_mechs and algorithm in sig_mechs:
        with oqs.Signature(algorithm) as sig:
            public_key = sig.generate_keypair()
            return algorithm, base64.b64encode(derive_key(public_key, 32)).decode()

    # 4) Clássicos
    if algorithm == "AES256_GCM":
        key = AESGCM.generate_key(bit_length=256)
        return algorithm, base64.b64encode(key).decode()
    if algorithm == "AES128_GCM":
        key = AESGCM.generate_key(bit_length=128)
        return algorithm, base64.b64encode(key).decode()
    if algorithm == "ChaCha20_Poly1305":
        key = ChaCha20Poly1305.generate_key()
        return algorithm, base64.b64encode(key).decode()

    raise ValueError(f"Algoritmo {algorithm} não suportado ou liboqs indisponível.")


def build_session(session_id: str, algorithm: str, ttl: int = 300):
    alg, key_material = generate_key(algorithm)
    expires = datetime.utcnow() + timedelta(seconds=ttl)
    return session_id, alg, key_material, expires