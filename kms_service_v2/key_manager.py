"""
Key Management Service (KMS) - Core Key Generation Module

Supports:
- Quantum Key Distribution (QKD) via quantum gateway simulation
- Post-Quantum Cryptography (PQC) via liboqs or simulation
- Classical cryptographic algorithms (AES, ChaCha20, RSA, ECC)
"""

import base64
import os
import time
from typing import Optional, Tuple
import uuid
from crypto_utils import derive_key
from quantum_gateway.gateway import generate_key_from_gateway
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives import serialization

# ====
# liboqs (Open Quantum Safe) - Primary PQC Backend
# ====
OQS_AVAILABLE = False
try:
    import oqs
    OQS_AVAILABLE = True
    print("[KMS] liboqs-python loaded successfully")
except ImportError:
    oqs = None
    OQS_AVAILABLE = False
    print("[KMS] liboqs-python not available — PQC will use simulation")

# ====
# pqcrypto - Fallback PQC Backend
# ====
PQC_AVAILABLE = False
try:
    import pqcrypto
    from pqcrypto.kem import ntruhrss701
    PQC_AVAILABLE = True
    print("[KMS] pqcrypto loaded successfully")
except ImportError:
    PQC_AVAILABLE = False
    print("[KMS] pqcrypto not available")


def _normalize_oqs_algorithm(requested: str, kem_mechs: list, sig_mechs: list) -> Tuple[Optional[str], str]:
    """Map requested algorithm name to available OQS mechanism."""
    if not requested:
        return None, ""

    if requested in kem_mechs:
        return requested, "kem"
    if requested in sig_mechs:
        return requested, "sig"

    aliases = {
        "Kyber512": "ML-KEM-512",
        "Kyber768": "ML-KEM-768",
        "Kyber1024": "ML-KEM-1024",
        "Dilithium2": "ML-DSA-44",
        "Dilithium3": "ML-DSA-65",
        "Dilithium5": "ML-DSA-87",
    }

    candidate = aliases.get(requested)
    if candidate:
        if candidate in kem_mechs:
            return candidate, "kem"
        if candidate in sig_mechs:
            return candidate, "sig"

    def normalize(s: str) -> str:
        return s.replace("-", "").replace("_", "").upper()

    rq = normalize(requested)
    for m in kem_mechs:
        if normalize(m) == rq:
            return m, "kem"
    for m in sig_mechs:
        if normalize(m) == rq:
            return m, "sig"

    return None, ""


def _oqs_kem_mechanisms():
    if not OQS_AVAILABLE:
        return []
    for fn_name in ["get_supported_KEM_mechanisms", "get_available_KEM_mechanisms",
                     "get_enabled_KEM_mechanisms", "get_enabled_KEMs"]:
        fn = getattr(oqs, fn_name, None)
        if fn:
            try:
                mechs = fn()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                continue
    return []


def _oqs_sig_mechanisms():
    if not OQS_AVAILABLE:
        return []
    for fn_name in ["get_supported_sig_mechanisms", "get_available_sig_mechanisms",
                     "get_enabled_sig_mechanisms", "get_enabled_Sigs"]:
        fn = getattr(oqs, fn_name, None)
        if fn:
            try:
                mechs = fn()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                continue
    return []


def _simulate_pqc_key(algorithm: str) -> Tuple[str, str]:
    """
    Simulate PQC key generation when liboqs/pqcrypto are not available.
    Generates cryptographically secure random bytes as key material.
    """
    # Key sizes matching real algorithms
    key_sizes = {
        "Kyber512": 32,
        "Kyber768": 32,
        "Kyber1024": 32,
        "ML-KEM-512": 32,
        "ML-KEM-768": 32,
        "ML-KEM-1024": 32,
        "Dilithium2": 32,
        "Dilithium3": 32,
        "Dilithium5": 32,
    }
    size = key_sizes.get(algorithm, 32)
    key = os.urandom(size)
    print(f"[KMS] Simulated PQC key generation for {algorithm} ({size} bytes)")
    return algorithm, base64.b64encode(key).decode()


def generate_key(algorithm: str) -> Tuple[str, str]:
    """
    Generate a session key using the requested algorithm.

    Priority:
      1. QKD (via Quantum Gateway) - for QKD_* algorithms
      2. PQC (liboqs) - primary
      3. PQC (pqcrypto) - fallback
      4. PQC (simulation) - when no PQC library available
      5. Classical (AES, ChaCha20, RSA, ECC)

    Returns:
        Tuple of (algorithm_used, key_material_base64)
    """
    # ---- QKD ----
    if algorithm.upper().startswith("QKD"):
        chosen_algo, key_material = generate_key_from_gateway(algorithm)
        if key_material:
            return chosen_algo, key_material
        else:
            print(f"[KMS] QKD requested ({algorithm}) but no material generated. "
                  f"Falling back to AES256_GCM.")
            key = AESGCM.generate_key(bit_length=256)
            return "AES256_GCM", base64.b64encode(key).decode()

    # ---- PQC via liboqs ----
    if OQS_AVAILABLE:
        kem_mechs = _oqs_kem_mechanisms()
        sig_mechs = _oqs_sig_mechanisms()
        normalized, cat = _normalize_oqs_algorithm(algorithm, kem_mechs, sig_mechs)

        if normalized and cat == "kem":
            try:
                with oqs.KeyEncapsulation(normalized) as kem:
                    public_key = kem.generate_keypair()
                    ciphertext, session_key = kem.encap_secret(public_key)
                    return normalized, base64.b64encode(session_key).decode()
            except Exception as e:
                print(f"[KMS Error] KEM {normalized} failed with liboqs: {e}")

        if normalized and cat == "sig":
            try:
                with oqs.Signature(normalized) as sig:
                    public_key = sig.generate_keypair()
                    return normalized, base64.b64encode(derive_key(public_key, 32)).decode()
            except Exception as e:
                print(f"[KMS Error] Signature {normalized} failed with liboqs: {e}")

    # ---- PQC via pqcrypto ----
    if PQC_AVAILABLE:
        pqc_kem_map = {
            "NTRU-HRSS-701": "pqcrypto.kem.ntruhrss701",
        }
        if algorithm in pqc_kem_map:
            try:
                module = __import__(pqc_kem_map[algorithm], fromlist=["generate_keypair", "encapsulate"])
                pk, sk = module.generate_keypair()
                ct, ss = module.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

    # ---- PQC Simulation (when no PQC library is available) ----
    pqc_names = ("KYBER", "ML-KEM", "DILITHIUM", "ML-DSA", "FALCON",
                 "SPHINCS", "NTRU", "SABER", "FRODO")
    if algorithm.upper().replace("-", "").replace("_", "").startswith(
            tuple(n.replace("-", "") for n in pqc_names)):
        return _simulate_pqc_key(algorithm)

    # ---- Classical ----
    classical_algorithms = {
        "AES256_GCM": lambda: AESGCM.generate_key(bit_length=256),
        "AES128_GCM": lambda: AESGCM.generate_key(bit_length=128),
        "ChaCha20_Poly1305": lambda: ChaCha20Poly1305.generate_key(),
        "3DES": lambda: os.urandom(24),
        "Blowfish": lambda: os.urandom(32),
    }
    if algorithm in classical_algorithms:
        key = classical_algorithms[algorithm]()
        return algorithm, base64.b64encode(key).decode()

    # RSA
    rsa_algorithms = {"RSA2048": 2048, "RSA4096": 4096}
    if algorithm in rsa_algorithms:
        key_size = rsa_algorithms[algorithm]
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # ECC
    ecc_algorithms = {
        "ECDH_P256": ec.SECP256R1(),
        "ECDH_P384": ec.SECP384R1(),
    }
    if algorithm in ecc_algorithms:
        curve = ecc_algorithms[algorithm]
        private_key = ec.generate_private_key(curve)
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    raise ValueError(f"Algorithm '{algorithm}' is not supported by the KMS")


def build_session(
    session_id: Optional[str],
    request_id: Optional[str],
    algorithm: str,
    ttl_seconds: int
) -> Tuple[str, str, str, str, int, bool, Optional[str], str]:
    """
    Build a complete key session.

    Returns:
        (session_id, request_id, selected_algorithm, key_material,
         expires_at, fallback_applied, fallback_reason, source_of_key)
    """
    selected_alg, key_material = generate_key(algorithm)

    fallback_applied = selected_alg != algorithm
    fallback_reason = None
    if fallback_applied:
        if algorithm.startswith("QKD"):
            fallback_reason = "QKD_UNAVAILABLE"
        elif algorithm.upper().replace("-", "").replace("_", "").startswith(
                ("KYBER", "MLKEM", "DILITHIUM", "FALCON", "SPHINCS", "NTRU", "SABER", "FRODO")):
            fallback_reason = "PQC_UNAVAILABLE"
        else:
            fallback_reason = "ALGO_NOT_SUPPORTED"

    # Determine key source
    if selected_alg.startswith("QKD"):
        source_of_key = "qkd"
    elif selected_alg.upper().replace("-", "").replace("_", "").startswith(
            ("KYBER", "MLKEM", "DILITHIUM", "MLDSA", "FALCON", "SPHINCS", "NTRU", "SABER", "FRODO")):
        source_of_key = "pqc"
    else:
        source_of_key = "classical"

    expires_at = int(time.time()) + int(ttl_seconds)
    sid = session_id or str(uuid.uuid4())
    rid = request_id or str(uuid.uuid4())

    return sid, rid, selected_alg, key_material, expires_at, fallback_applied, fallback_reason, source_of_key


def get_supported_algorithms():
    """Return all algorithms supported by the KMS."""
    supported = {
        "classical": [
            "AES256_GCM", "AES128_GCM", "ChaCha20_Poly1305",
            "RSA2048", "RSA4096",
            "ECDH_P256", "ECDH_P384",
            "3DES", "Blowfish",
        ],
        "pqc_kems": ["Kyber512", "Kyber768", "Kyber1024"],
        "pqc_signatures": [],
        "qkd": [
            "QKD_BB84", "QKD_E91", "QKD_CV", "QKD_MDI",
            "QKD_SARG04", "QKD_DecoyState", "QKD_DI",
        ],
        "oqs_available": OQS_AVAILABLE,
        "pqcrypto_available": PQC_AVAILABLE,
    }

    if OQS_AVAILABLE:
        supported["oqs_kems"] = _oqs_kem_mechanisms()
        supported["oqs_signatures"] = _oqs_sig_mechanisms()

    return supported


def get_algorithm_info(algorithm: str):
    """Return detailed information about a specific algorithm."""
    info = {
        "algorithm": algorithm,
        "category": "unknown",
        "security_level": "unknown",
        "key_size_bits": 256,
        "quantum_resistant": False,
        "recommended": True,
    }

    if algorithm in ["AES256_GCM", "ChaCha20_Poly1305"]:
        info.update({"category": "symmetric", "security_level": "high",
                      "quantum_resistant": False, "recommended": True})
    elif algorithm == "AES128_GCM":
        info.update({"category": "symmetric", "security_level": "medium",
                      "key_size_bits": 128, "quantum_resistant": False})
    elif algorithm.startswith("RSA"):
        key_size = int(algorithm.replace("RSA", ""))
        info.update({"category": "asymmetric", "key_size_bits": key_size,
                      "security_level": "high" if key_size >= 4096 else "medium",
                      "quantum_resistant": False})
    elif algorithm.startswith("ECDH"):
        info.update({"category": "asymmetric", "security_level": "high",
                      "quantum_resistant": False})
    elif algorithm.startswith(("Kyber", "ML-KEM", "NTRU", "Saber", "Frodo")):
        info.update({"category": "pqc_kem", "security_level": "high",
                      "quantum_resistant": True})
    elif algorithm.startswith(("Dilithium", "ML-DSA", "Falcon", "SPHINCS")):
        info.update({"category": "pqc_signature", "security_level": "high",
                      "quantum_resistant": True})
    elif algorithm.startswith("QKD"):
        info.update({"category": "qkd", "security_level": "maximum",
                      "quantum_resistant": True})
    elif algorithm in ["3DES", "Blowfish"]:
        info.update({"category": "legacy", "security_level": "low",
                      "quantum_resistant": False, "recommended": False})

    return info
