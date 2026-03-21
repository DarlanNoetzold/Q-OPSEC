import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


CLASSICAL_SYMMETRIC_ALGORITHMS = {
    "AES256_GCM": {"key_bits": 256, "category": "symmetric", "security_level": "high"},
    "AES128_GCM": {"key_bits": 128, "category": "symmetric", "security_level": "medium"},
    "ChaCha20_Poly1305": {"key_bits": 256, "category": "symmetric", "security_level": "high"},
    "3DES": {"key_bits": 192, "category": "symmetric_legacy", "security_level": "low"},
    "Blowfish": {"key_bits": 256, "category": "symmetric_legacy", "security_level": "low"},
}

CLASSICAL_ASYMMETRIC_ALGORITHMS = {
    "RSA2048": {"key_bits": 2048, "category": "asymmetric", "security_level": "medium"},
    "RSA4096": {"key_bits": 4096, "category": "asymmetric", "security_level": "high"},
    "ECDH_P256": {"curve": "P-256", "category": "asymmetric", "security_level": "high"},
    "ECDH_P384": {"curve": "P-384", "category": "asymmetric", "security_level": "high"},
    "ECDH_P521": {"curve": "P-521", "category": "asymmetric", "security_level": "high"},
}


def list_classical_algorithms():
    all_algorithms = {}
    for name, meta in CLASSICAL_SYMMETRIC_ALGORITHMS.items():
        all_algorithms[name] = {
            **meta,
            "quantum_resistant": False,
            "recommended": meta["security_level"] != "low",
        }
    for name, meta in CLASSICAL_ASYMMETRIC_ALGORITHMS.items():
        all_algorithms[name] = {
            **meta,
            "quantum_resistant": False,
            "recommended": True,
        }
    return all_algorithms


def derive_session_key(raw_material: bytes, length: int = 32, context: bytes = b"qopsec-session-key") -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=context,
    )
    return hkdf.derive(raw_material)


def generate_symmetric_key(algorithm: str) -> bytes:
    if algorithm == "AES256_GCM":
        return AESGCM.generate_key(bit_length=256)
    if algorithm == "AES128_GCM":
        return AESGCM.generate_key(bit_length=128)
    if algorithm == "ChaCha20_Poly1305":
        return ChaCha20Poly1305.generate_key()
    if algorithm == "3DES":
        return os.urandom(24)
    if algorithm == "Blowfish":
        return os.urandom(32)
    raise ValueError(f"Unknown symmetric algorithm: {algorithm}")


def generate_asymmetric_key(algorithm: str) -> bytes:
    if algorithm.startswith("RSA"):
        key_size = int(algorithm.replace("RSA", ""))
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return derive_session_key(key_bytes, 32)

    curve_map = {
        "ECDH_P256": ec.SECP256R1(),
        "ECDH_P384": ec.SECP384R1(),
        "ECDH_P521": ec.SECP521R1(),
    }

    if algorithm in curve_map:
        private_key = ec.generate_private_key(curve_map[algorithm])
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return derive_session_key(key_bytes, 32)

    raise ValueError(f"Unknown asymmetric algorithm: {algorithm}")


def generate_classical_key(algorithm: str) -> str:
    if algorithm in CLASSICAL_SYMMETRIC_ALGORITHMS:
        raw_key = generate_symmetric_key(algorithm)
        return base64.b64encode(raw_key).decode()

    if algorithm in CLASSICAL_ASYMMETRIC_ALGORITHMS:
        derived_key = generate_asymmetric_key(algorithm)
        return base64.b64encode(derived_key).decode()

    raise ValueError(f"Unsupported classical algorithm: {algorithm}")


def is_classical_algorithm(algorithm: str) -> bool:
    return algorithm in CLASSICAL_SYMMETRIC_ALGORITHMS or algorithm in CLASSICAL_ASYMMETRIC_ALGORITHMS
