"""
Key Management Service (KMS) - Core Key Generation Module

This module provides secure key generation capabilities for:
- Quantum Key Distribution (QKD) via quantum gateway
- Post-Quantum Cryptography (PQC) via liboqs and pqcrypto
- Classical cryptographic algorithms (AES, RSA, ECC, etc.)

The module implements a priority-based fallback system to ensure
key generation always succeeds with the best available algorithm.
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
from datetime import datetime, timedelta
# ====================================================================
# liboqs (Open Quantum Safe) - Primary PQC Backend
# ====================================================================
OQS_AVAILABLE = False
try:
    import oqs

    OQS_AVAILABLE = True
    print("[KMS] liboqs-python loaded successfully")
except ImportError:
    oqs = None
    OQS_AVAILABLE = False
    print("[KMS] liboqs-python not available")

# ====================================================================
# pqcrypto - Fallback PQC Backend
# ====================================================================
PQC_AVAILABLE = False
try:
    import pqcrypto

    # Verify if KEMs are actually available
    try:
        from pqcrypto.kem import ntruhrss701

        PQC_AVAILABLE = True
        print("[KMS] pqcrypto loaded successfully")
    except ImportError:
        print("[KMS] pqcrypto installed but no KEMs available")
        PQC_AVAILABLE = False
except ImportError:
    print("[KMS] pqcrypto not available")
    PQC_AVAILABLE = False


def _normalize_oqs_algorithm(requested: str, kem_mechs: list, sig_mechs: list) -> Tuple[Optional[str], str]:
    """
    Map requested algorithm name to available OQS mechanism.

    This function handles common aliases and naming variations,
    particularly for post-NIST standardized algorithms.

    Args:
        requested: Algorithm name as requested by user
        kem_mechs: List of available KEM mechanisms from liboqs
        sig_mechs: List of available signature mechanisms from liboqs

    Returns:
        Tuple of (normalized_name_or_None, category: 'kem'|'sig'|'')
    """
    if not requested:
        return None, ""

    # Try exact match first
    if requested in kem_mechs:
        return requested, "kem"
    if requested in sig_mechs:
        return requested, "sig"

    # Common aliases (post-NIST standardization)
    aliases = {
        # Kyber -> ML-KEM (NIST standardized names)
        "Kyber512": "ML-KEM-512",
        "Kyber768": "ML-KEM-768",
        "Kyber1024": "ML-KEM-1024",

        # Dilithium -> ML-DSA (NIST standardized names)
        "Dilithium2": "ML-DSA-44",
        "Dilithium3": "ML-DSA-65",
        "Dilithium5": "ML-DSA-87",

        # Falcon variants
        "Falcon-512": "Falcon-512",
        "Falcon-1024": "Falcon-1024",
        "FALCON-512": "Falcon-512",
        "FALCON-1024": "Falcon-1024",
    }

    candidate = aliases.get(requested)
    if candidate:
        if candidate in kem_mechs:
            return candidate, "kem"
        if candidate in sig_mechs:
            return candidate, "sig"

    # Case-insensitive and hyphen-tolerant matching
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
    """
    Detect available KEM mechanisms in liboqs (multi-version compatibility).

    Different versions of liboqs use different function names to list
    available mechanisms. This function tries all known variants.

    Returns:
        List of available KEM mechanism names
    """
    if not OQS_AVAILABLE:
        return []

    # Try different function names across liboqs versions
    function_names = [
        "get_supported_KEM_mechanisms",
        "get_available_KEM_mechanisms",
        "get_enabled_KEM_mechanisms",
        "get_enabled_KEMs"
    ]

    for fn_name in function_names:
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
    """
    Detect available signature mechanisms in liboqs (multi-version compatibility).

    Returns:
        List of available signature mechanism names
    """
    if not OQS_AVAILABLE:
        return []

    # Try different function names across liboqs versions
    function_names = [
        "get_supported_sig_mechanisms",
        "get_available_sig_mechanisms",
        "get_enabled_sig_mechanisms",
        "get_enabled_Sigs"
    ]

    for fn_name in function_names:
        fn = getattr(oqs, fn_name, None)
        if fn:
            try:
                mechs = fn()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                continue
    return []


def _get_pqcrypto_kems():
    """
    Detect actually available KEM algorithms in pqcrypto.

    Returns:
        List of available KEM algorithm names
    """
    if not PQC_AVAILABLE:
        return []

    available_kems = []
    kem_modules = [
        ('ntruhrss701', 'NTRU-HRSS-701'),
        ('ntruhps2048509', 'NTRU-HPS-2048-509'),
        ('ntruhps2048677', 'NTRU-HPS-2048-677'),
        ('lightsaber', 'LightSaber'),
        ('saber', 'Saber'),
        ('firesaber', 'FireSaber'),
        ('frodokem640aes', 'FrodoKEM-640-AES'),
        ('frodokem976aes', 'FrodoKEM-976-AES'),
        ('frodokem1344aes', 'FrodoKEM-1344-AES')
    ]

    for module_name, algo_name in kem_modules:
        try:
            __import__(f'pqcrypto.kem.{module_name}')
            available_kems.append(algo_name)
        except ImportError:
            continue

    return available_kems


def _get_pqcrypto_sigs():
    """
    Detect actually available signature algorithms in pqcrypto.

    Returns:
        List of available signature algorithm names
    """
    if not PQC_AVAILABLE:
        return []

    available_sigs = []
    sig_modules = [
        ('dilithium2', 'Dilithium2'),
        ('dilithium3', 'Dilithium3'),
        ('dilithium5', 'Dilithium5'),
        ('falcon512', 'Falcon-512'),
        ('falcon1024', 'Falcon-1024'),
        ('sphincssha256128ssimple', 'SPHINCS+-SHA256-128s'),
        ('sphincssha256192ssimple', 'SPHINCS+-SHA256-192s'),
        ('sphincssha256256ssimple', 'SPHINCS+-SHA256-256s')
    ]

    for module_name, algo_name in sig_modules:
        try:
            __import__(f'pqcrypto.sign.{module_name}')
            available_sigs.append(algo_name)
        except ImportError:
            continue

    return available_sigs


def generate_key(algorithm: str) -> Tuple[str, str]:
    """
    Generate a secure session key using the requested algorithm.

    Priority order:
      1. QKD (via Quantum Gateway) - only for QKD_* algorithms
      2. PQC (liboqs - primary backend)
      3. PQC (pqcrypto - fallback backend)
      4. Classical cryptography (AES, ChaCha20, RSA, ECC)
      5. Automatic fallback to AES256_GCM if QKD is requested but unavailable

    Args:
        algorithm: Name of the cryptographic algorithm to use

    Returns:
        Tuple of (algorithm_used, key_material_base64)

    Raises:
        ValueError: If the algorithm is not supported by any backend
    """

    # ====================================================================
    # 1. QKD (Quantum Key Distribution) - Only for QKD_* algorithms
    # ====================================================================
    if algorithm.upper().startswith("QKD"):
        chosen_algo, key_material = generate_key_from_gateway(algorithm)
        if key_material:
            return chosen_algo, key_material
        else:
            # Clear fallback if QKD gateway failed
            print(f"[KMS] QKD requested ({algorithm}) but no material was generated. "
                  f"Falling back to AES256_GCM.")
            key = AESGCM.generate_key(bit_length=256)
            return "AES256_GCM", base64.b64encode(key).decode()

    # ====================================================================
    # 2. Post-Quantum Cryptography via liboqs (Primary PQC Backend)
    # ====================================================================
    if OQS_AVAILABLE:
        kem_mechs = _oqs_kem_mechanisms()
        sig_mechs = _oqs_sig_mechanisms()
        normalized, cat = _normalize_oqs_algorithm(algorithm, kem_mechs, sig_mechs)

        # Handle KEM (Key Encapsulation Mechanism)
        if normalized and cat == "kem":
            try:
                with oqs.KeyEncapsulation(normalized) as kem:
                    public_key = kem.generate_keypair()
                    ciphertext, session_key = kem.encap_secret(public_key)
                    return normalized, base64.b64encode(session_key).decode()
            except Exception as e:
                print(f"[KMS Error] Failed to generate KEM {normalized} with liboqs: {e}")

        # Handle Digital Signatures
        if normalized and cat == "sig":
            try:
                with oqs.Signature(normalized) as sig:
                    public_key = sig.generate_keypair()
                    # Derive session key from public key material
                    return normalized, base64.b64encode(derive_key(public_key, 32)).decode()
            except Exception as e:
                print(f"[KMS Error] Failed to generate Signature {normalized} with liboqs: {e}")

    # ====================================================================
    # 3. Post-Quantum Cryptography via pqcrypto (Fallback PQC Backend)
    # ====================================================================
    if PQC_AVAILABLE:
        # KEM algorithm mapping
        pqc_kem_map = {
            "NTRU-HRSS-701": ("pqcrypto.kem.ntruhrss701", "ntruhrss701"),
            "NTRU-HPS-2048-509": ("pqcrypto.kem.ntruhps2048509", "ntruhps2048509"),
            "NTRU-HPS-2048-677": ("pqcrypto.kem.ntruhps2048677", "ntruhps2048677"),
            "LightSaber": ("pqcrypto.kem.lightsaber", "lightsaber"),
            "Saber": ("pqcrypto.kem.saber", "saber"),
            "FireSaber": ("pqcrypto.kem.firesaber", "firesaber"),
            "FrodoKEM-640-AES": ("pqcrypto.kem.frodokem640aes", "frodokem640aes"),
            "FrodoKEM-976-AES": ("pqcrypto.kem.frodokem976aes", "frodokem976aes"),
            "FrodoKEM-1344-AES": ("pqcrypto.kem.frodokem1344aes", "frodokem1344aes")
        }

        # Signature algorithm mapping
        pqc_sig_map = {
            "Dilithium2": ("pqcrypto.sign.dilithium2", "dilithium2"),
            "Dilithium3": ("pqcrypto.sign.dilithium3", "dilithium3"),
            "Dilithium5": ("pqcrypto.sign.dilithium5", "dilithium5"),
            "Falcon-512": ("pqcrypto.sign.falcon512", "falcon512"),
            "Falcon-1024": ("pqcrypto.sign.falcon1024", "falcon1024"),
            "SPHINCS+-SHA256-128s": ("pqcrypto.sign.sphincssha256128ssimple", "sphincssha256128ssimple"),
            "SPHINCS+-SHA256-192s": ("pqcrypto.sign.sphincssha256192ssimple", "sphincssha256192ssimple"),
            "SPHINCS+-SHA256-256s": ("pqcrypto.sign.sphincssha256256ssimple", "sphincssha256256ssimple")
        }

        # Try KEM algorithms
        if algorithm in pqc_kem_map:
            module_path, module_name = pqc_kem_map[algorithm]
            try:
                module = __import__(module_path, fromlist=[module_name])
                pk, sk = module.generate_keypair()
                ct, ss = module.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                print(f"[KMS Warning] pqcrypto module {module_path} not available")

        # Try signature algorithms
        if algorithm in pqc_sig_map:
            module_path, module_name = pqc_sig_map[algorithm]
            try:
                module = __import__(module_path, fromlist=[module_name])
                pk, sk = module.generate_keypair()
                # Derive session key from public key material
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                print(f"[KMS Warning] pqcrypto module {module_path} not available")

    # ====================================================================
    # 4. Classical Cryptographic Algorithms (Always Available)
    # ====================================================================

    # Symmetric encryption algorithms
    classical_algorithms = {
        # AES family (recommended)
        "AES256_GCM": lambda: AESGCM.generate_key(bit_length=256),
        "AES128_GCM": lambda: AESGCM.generate_key(bit_length=128),

        # ChaCha20-Poly1305 (modern alternative to AES)
        "ChaCha20_Poly1305": lambda: ChaCha20Poly1305.generate_key(),

        # Legacy algorithms (not recommended for new applications)
        "3DES": lambda: os.urandom(24),  # 3DES uses 192 bits (24 bytes)
        "Blowfish": lambda: os.urandom(32),  # Blowfish supports up to 448 bits
    }

    if algorithm in classical_algorithms:
        key = classical_algorithms[algorithm]()
        return algorithm, base64.b64encode(key).decode()

    # RSA key generation (derive session key from private key)
    rsa_algorithms = {
        "RSA2048": 2048,
        "RSA4096": 4096
    }

    if algorithm in rsa_algorithms:
        key_size = rsa_algorithms[algorithm]
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        # Derive 256-bit session key from RSA private key
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # Elliptic Curve Cryptography (ECC)
    ecc_algorithms = {
        "ECDH_P256": ec.SECP256R1(),
        "ECDH_P384": ec.SECP384R1(),
        "ECDH_Curve25519": ec.X25519()
    }

    if algorithm in ecc_algorithms:
        curve = ecc_algorithms[algorithm]
        private_key = ec.generate_private_key(curve)

        if algorithm == "ECDH_Curve25519":
            # X25519 uses raw encoding
            key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            # Other curves use DER encoding
            key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        # Derive 256-bit session key from ECC private key
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # ====================================================================
    # 5. Algorithm Not Supported
    # ====================================================================
    raise ValueError(f"Algorithm '{algorithm}' is not supported by the KMS")


def build_session(
    session_id: Optional[str],
    request_id: str,  # <- NOVO: Adicionar request_id aqui
    algorithm: str,
    ttl_seconds: int
) -> Tuple[str, str, str, int, bool, Optional[str], str, str]:
    """
    Build a complete key session with metadata.

    Args:
    session_id: Optional session identifier (generated if None)
    request_id: Unique identifier for the original request (from Context API) # <- NOVO
    algorithm: Requested cryptographic algorithm
    ttl_seconds: Time-to-live for the session in seconds

    Returns:
    Tuple containing:
    - session_id: Unique session identifier
    - request_id: Unique request identifier # <- NOVO
    - selected_algorithm: Actually used algorithm (may differ due to fallback)
    - key_material: Base64-encoded key material
    - expires_at: Unix timestamp when session expires
    - fallback_applied: Whether fallback was used
    - fallback_reason: Reason for fallback (if any)
    - source_of_key: Source category ('qkd', 'pqc', 'classical')
    """
    # Generate the key
    selected_alg, key_material = generate_key(algorithm)

    # Detect if fallback was applied
    fallback_applied = selected_alg != algorithm
    fallback_reason = None
    if fallback_applied:
        if algorithm.startswith("QKD"):
            fallback_reason = "QKD_UNAVAILABLE"
        elif algorithm.upper().startswith(
            ("KYBER", "ML-KEM", "DILITHIUM", "FALCON", "SPHINCS", "NTRU", "SABER", "FRODO")):
            fallback_reason = "PQC_UNAVAILABLE"
        else:
            fallback_reason = "ALGO_NOT_SUPPORTED"

    # Determine key source for telemetry
    if selected_alg.startswith("QKD"):
        source_of_key = "qkd"
    elif selected_alg.upper().startswith(
        ("KYBER", "ML-KEM", "DILITHIUM", "FALCON", "SPHINCS", "NTRU", "SABER", "FRODO")):
        source_of_key = "pqc"
    else:
        source_of_key = "classical"

    # Calculate expiration timestamp
    expires_at = int(time.time()) + int(ttl_seconds)
    sid = session_id or str(uuid.uuid4())

    return sid, request_id, selected_alg, key_material, expires_at, fallback_applied, fallback_reason, source_of_key


def get_supported_algorithms():
    """
    Return comprehensive list of all algorithms supported by the KMS.

    Returns:
        Dictionary with algorithm categories and their supported algorithms
    """
    supported = {
        "classical": [
            "AES256_GCM", "AES128_GCM", "ChaCha20_Poly1305",
            "RSA2048", "RSA4096",
            "ECDH_P256", "ECDH_P384", "ECDH_Curve25519",
            "3DES", "Blowfish"  # Legacy algorithms
        ],
        "pqc_kems": [],
        "pqc_signatures": [],
        "qkd": [],
        "oqs_kems": [],
        "oqs_signatures": []
    }

    # Add liboqs algorithms (primary PQC backend)
    if OQS_AVAILABLE:
        supported["oqs_kems"] = _oqs_kem_mechanisms()
        supported["oqs_signatures"] = _oqs_sig_mechanisms()

    # Add pqcrypto algorithms (fallback PQC backend)
    if PQC_AVAILABLE:
        supported["pqc_kems"] = _get_pqcrypto_kems()
        supported["pqc_signatures"] = _get_pqcrypto_sigs()

    # Add QKD algorithms (if available)
    qkd_available = os.getenv("QKD_AVAILABLE", "false").lower() == "true"
    if qkd_available:
        supported["qkd"] = [
            "QKD_BB84", "QKD_E91", "QKD_CV", "QKD_MDI",
            "QKD_SARG04", "QKD_DecoyState", "QKD_DI"
        ]

    return supported


def get_algorithm_info(algorithm: str):
    """
    Return detailed information about a specific algorithm.

    Args:
        algorithm: Name of the algorithm to get information about

    Returns:
        Dictionary with algorithm metadata including security level,
        quantum resistance, and recommendations
    """
    info = {
        "algorithm": algorithm,
        "category": "unknown",
        "security_level": "unknown",
        "key_size_bits": 256,  # Default: 256 bits
        "quantum_resistant": False,
        "recommended": True
    }

    # Classical symmetric algorithms
    if algorithm in ["AES256_GCM", "ChaCha20_Poly1305"]:
        info.update({
            "category": "symmetric",
            "security_level": "high",
            "quantum_resistant": False,
            "recommended": True
        })
    elif algorithm == "AES128_GCM":
        info.update({
            "category": "symmetric",
            "security_level": "medium",
            "key_size_bits": 128,
            "quantum_resistant": False,
            "recommended": True
        })

    # Classical asymmetric algorithms
    elif algorithm.startswith("RSA"):
        key_size = int(algorithm.replace("RSA", ""))
        info.update({
            "category": "asymmetric",
            "security_level": "high" if key_size >= 4096 else "medium",
            "key_size_bits": key_size,
            "quantum_resistant": False,
            "recommended": key_size >= 2048
        })
    elif algorithm.startswith("ECDH"):
        info.update({
            "category": "asymmetric",
            "security_level": "high",
            "quantum_resistant": False,
            "recommended": True
        })

    # Post-Quantum KEM algorithms
    elif algorithm.startswith(("Kyber", "ML-KEM", "NTRU", "Saber", "Frodo")):
        info.update({
            "category": "pqc_kem",
            "security_level": "high",
            "quantum_resistant": True,
            "recommended": True
        })

    # Post-Quantum signature algorithms
    elif algorithm.startswith(("Dilithium", "ML-DSA", "Falcon", "SPHINCS")):
        info.update({
            "category": "pqc_signature",
            "security_level": "high",
            "quantum_resistant": True,
            "recommended": True
        })

    # Quantum Key Distribution
    elif algorithm.startswith("QKD"):
        info.update({
            "category": "qkd",
            "security_level": "maximum",
            "quantum_resistant": True,
            "recommended": True
        })

    # Legacy algorithms (not recommended)
    elif algorithm in ["3DES", "Blowfish"]:
        info.update({
            "category": "legacy",
            "security_level": "low",
            "quantum_resistant": False,
            "recommended": False
        })

    return info