import base64
import os
import logging
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger("qopsec.pqc")

OQS_AVAILABLE = False
try:
    import oqs
    OQS_AVAILABLE = True
except ImportError:
    oqs = None

PQC_AVAILABLE = False
try:
    import pqcrypto
    PQC_AVAILABLE = True
except ImportError:
    pqcrypto = None


PQC_KEM_ALGORITHMS = {
    "Kyber512": {
        "oqs_name": "Kyber512",
        "aliases": ["ML-KEM-512"],
        "family": "lattice",
        "security_level": 1,
        "key_size_bits": 256,
    },
    "Kyber768": {
        "oqs_name": "Kyber768",
        "aliases": ["ML-KEM-768"],
        "family": "lattice",
        "security_level": 3,
        "key_size_bits": 256,
    },
    "Kyber1024": {
        "oqs_name": "Kyber1024",
        "aliases": ["ML-KEM-1024"],
        "family": "lattice",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "NTRU-HPS-2048-509": {
        "oqs_name": "NTRU-HPS-2048-509",
        "aliases": ["NTRU-HPS-2048-509"],
        "family": "lattice",
        "security_level": 1,
        "key_size_bits": 256,
    },
    "NTRU-HPS-2048-677": {
        "oqs_name": "NTRU-HPS-2048-677",
        "aliases": ["NTRU-HPS-2048-677"],
        "family": "lattice",
        "security_level": 3,
        "key_size_bits": 256,
    },
    "NTRU-HPS-4096-821": {
        "oqs_name": "NTRU-HPS-4096-821",
        "aliases": ["NTRU-HPS-4096-821"],
        "family": "lattice",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "NTRU-HRSS-701": {
        "oqs_name": "NTRU-HRSS-701",
        "aliases": ["NTRU-HRSS-701"],
        "family": "lattice",
        "security_level": 3,
        "key_size_bits": 256,
    },
    "LightSaber": {
        "oqs_name": "LightSaber-KEM",
        "aliases": ["SABER-Light", "LightSaber"],
        "family": "lattice",
        "security_level": 1,
        "key_size_bits": 256,
    },
    "Saber": {
        "oqs_name": "Saber-KEM",
        "aliases": ["SABER"],
        "family": "lattice",
        "security_level": 3,
        "key_size_bits": 256,
    },
    "FireSaber": {
        "oqs_name": "FireSaber-KEM",
        "aliases": ["SABER-Fire", "FireSaber"],
        "family": "lattice",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "FrodoKEM-640-AES": {
        "oqs_name": "FrodoKEM-640-AES",
        "aliases": ["FrodoKEM-640-AES"],
        "family": "lattice",
        "security_level": 1,
        "key_size_bits": 128,
    },
    "FrodoKEM-976-AES": {
        "oqs_name": "FrodoKEM-976-AES",
        "aliases": ["FrodoKEM-976-AES"],
        "family": "lattice",
        "security_level": 3,
        "key_size_bits": 192,
    },
    "FrodoKEM-1344-AES": {
        "oqs_name": "FrodoKEM-1344-AES",
        "aliases": ["FrodoKEM-1344-AES"],
        "family": "lattice",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "FrodoKEM-640-SHAKE": {
        "oqs_name": "FrodoKEM-640-SHAKE",
        "aliases": ["FrodoKEM-640-SHAKE"],
        "family": "lattice",
        "security_level": 1,
        "key_size_bits": 128,
    },
    "FrodoKEM-976-SHAKE": {
        "oqs_name": "FrodoKEM-976-SHAKE",
        "aliases": ["FrodoKEM-976-SHAKE"],
        "family": "lattice",
        "security_level": 3,
        "key_size_bits": 192,
    },
    "FrodoKEM-1344-SHAKE": {
        "oqs_name": "FrodoKEM-1344-SHAKE",
        "aliases": ["FrodoKEM-1344-SHAKE"],
        "family": "lattice",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "Classic-McEliece-348864": {
        "oqs_name": "Classic-McEliece-348864",
        "aliases": ["McEliece-348864"],
        "family": "code-based",
        "security_level": 1,
        "key_size_bits": 256,
    },
    "Classic-McEliece-460896": {
        "oqs_name": "Classic-McEliece-460896",
        "aliases": ["McEliece-460896"],
        "family": "code-based",
        "security_level": 3,
        "key_size_bits": 256,
    },
    "Classic-McEliece-6688128": {
        "oqs_name": "Classic-McEliece-6688128",
        "aliases": ["McEliece-6688128"],
        "family": "code-based",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "Classic-McEliece-6960119": {
        "oqs_name": "Classic-McEliece-6960119",
        "aliases": ["McEliece-6960119"],
        "family": "code-based",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "Classic-McEliece-8192128": {
        "oqs_name": "Classic-McEliece-8192128",
        "aliases": ["McEliece-8192128"],
        "family": "code-based",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "HQC-128": {
        "oqs_name": "HQC-128",
        "aliases": ["HQC128"],
        "family": "code-based",
        "security_level": 1,
        "key_size_bits": 128,
    },
    "HQC-192": {
        "oqs_name": "HQC-192",
        "aliases": ["HQC192"],
        "family": "code-based",
        "security_level": 3,
        "key_size_bits": 192,
    },
    "HQC-256": {
        "oqs_name": "HQC-256",
        "aliases": ["HQC256"],
        "family": "code-based",
        "security_level": 5,
        "key_size_bits": 256,
    },
    "BIKE-L1": {
        "oqs_name": "BIKE-L1",
        "aliases": ["BIKE1"],
        "family": "code-based",
        "security_level": 1,
        "key_size_bits": 128,
    },
    "BIKE-L3": {
        "oqs_name": "BIKE-L3",
        "aliases": ["BIKE3"],
        "family": "code-based",
        "security_level": 3,
        "key_size_bits": 192,
    },
    "BIKE-L5": {
        "oqs_name": "BIKE-L5",
        "aliases": ["BIKE5"],
        "family": "code-based",
        "security_level": 5,
        "key_size_bits": 256,
    },
}

PQC_SIGNATURE_ALGORITHMS = {
    "Dilithium2": {
        "oqs_name": "Dilithium2",
        "aliases": ["ML-DSA-44"],
        "family": "lattice",
        "security_level": 2,
    },
    "Dilithium3": {
        "oqs_name": "Dilithium3",
        "aliases": ["ML-DSA-65"],
        "family": "lattice",
        "security_level": 3,
    },
    "Dilithium5": {
        "oqs_name": "Dilithium5",
        "aliases": ["ML-DSA-87"],
        "family": "lattice",
        "security_level": 5,
    },
    "Falcon-512": {
        "oqs_name": "Falcon-512",
        "aliases": ["FALCON-512"],
        "family": "lattice",
        "security_level": 1,
    },
    "Falcon-1024": {
        "oqs_name": "Falcon-1024",
        "aliases": ["FALCON-1024"],
        "family": "lattice",
        "security_level": 5,
    },
    "SPHINCS+-SHA2-128f-simple": {
        "oqs_name": "SPHINCS+-SHA2-128f-simple",
        "aliases": ["SPHINCS+-SHA256-128f"],
        "family": "hash-based",
        "security_level": 1,
    },
    "SPHINCS+-SHA2-192f-simple": {
        "oqs_name": "SPHINCS+-SHA2-192f-simple",
        "aliases": ["SPHINCS+-SHA256-192f"],
        "family": "hash-based",
        "security_level": 3,
    },
    "SPHINCS+-SHA2-256f-simple": {
        "oqs_name": "SPHINCS+-SHA2-256f-simple",
        "aliases": ["SPHINCS+-SHA256-256f"],
        "family": "hash-based",
        "security_level": 5,
    },
    "SPHINCS+-SHAKE-128f-simple": {
        "oqs_name": "SPHINCS+-SHAKE-128f-simple",
        "aliases": ["SPHINCS+-SHAKE-128f"],
        "family": "hash-based",
        "security_level": 1,
    },
    "SPHINCS+-SHAKE-256f-simple": {
        "oqs_name": "SPHINCS+-SHAKE-256f-simple",
        "aliases": ["SPHINCS+-SHAKE-256f"],
        "family": "hash-based",
        "security_level": 5,
    },
}


def _get_oqs_kem_mechanisms() -> List[str]:
    if not OQS_AVAILABLE:
        return []
    for function_name in [
        "get_supported_KEM_mechanisms",
        "get_available_KEM_mechanisms",
        "get_enabled_KEM_mechanisms",
        "get_enabled_KEMs",
    ]:
        fn = getattr(oqs, function_name, None)
        if fn:
            try:
                result = fn()
                if isinstance(result, (list, tuple)):
                    return list(result)
            except Exception:
                continue
    return []


def _get_oqs_sig_mechanisms() -> List[str]:
    if not OQS_AVAILABLE:
        return []
    for function_name in [
        "get_supported_sig_mechanisms",
        "get_available_sig_mechanisms",
        "get_enabled_sig_mechanisms",
        "get_enabled_Sigs",
    ]:
        fn = getattr(oqs, function_name, None)
        if fn:
            try:
                result = fn()
                if isinstance(result, (list, tuple)):
                    return list(result)
            except Exception:
                continue
    return []


def _resolve_oqs_name(requested: str) -> Optional[Tuple[str, str]]:
    available_kems = _get_oqs_kem_mechanisms()
    available_sigs = _get_oqs_sig_mechanisms()

    for canonical, meta in PQC_KEM_ALGORITHMS.items():
        if requested == canonical or requested in meta["aliases"] or requested == meta["oqs_name"]:
            if meta["oqs_name"] in available_kems:
                return meta["oqs_name"], "kem"
            for alt in [canonical] + meta["aliases"]:
                if alt in available_kems:
                    return alt, "kem"

    for canonical, meta in PQC_SIGNATURE_ALGORITHMS.items():
        if requested == canonical or requested in meta["aliases"] or requested == meta["oqs_name"]:
            if meta["oqs_name"] in available_sigs:
                return meta["oqs_name"], "sig"
            for alt in [canonical] + meta["aliases"]:
                if alt in available_sigs:
                    return alt, "sig"

    def normalized(s):
        return s.replace("-", "").replace("_", "").replace(" ", "").upper()

    requested_normalized = normalized(requested)

    for mech in available_kems:
        if normalized(mech) == requested_normalized:
            return mech, "kem"

    for mech in available_sigs:
        if normalized(mech) == requested_normalized:
            return mech, "sig"

    return None


def _generate_via_oqs_kem(mechanism_name: str) -> bytes:
    with oqs.KeyEncapsulation(mechanism_name) as kem:
        public_key = kem.generate_keypair()
        _ciphertext, shared_secret = kem.encap_secret(public_key)
        return shared_secret


def _generate_via_oqs_sig(mechanism_name: str) -> bytes:
    from crypto.classical import derive_session_key
    with oqs.Signature(mechanism_name) as sig:
        public_key = sig.generate_keypair()
        return derive_session_key(public_key, 32)


def _simulate_pqc_key(algorithm: str) -> bytes:
    kem_meta = PQC_KEM_ALGORITHMS.get(algorithm)
    if kem_meta:
        key_bytes = kem_meta["key_size_bits"] // 8
        return os.urandom(max(key_bytes, 16))

    sig_meta = PQC_SIGNATURE_ALGORITHMS.get(algorithm)
    if sig_meta:
        return os.urandom(32)

    return os.urandom(32)


def is_pqc_algorithm(algorithm: str) -> bool:
    if algorithm in PQC_KEM_ALGORITHMS or algorithm in PQC_SIGNATURE_ALGORITHMS:
        return True

    for meta in PQC_KEM_ALGORITHMS.values():
        if algorithm in meta["aliases"] or algorithm == meta["oqs_name"]:
            return True
    for meta in PQC_SIGNATURE_ALGORITHMS.values():
        if algorithm in meta["aliases"] or algorithm == meta["oqs_name"]:
            return True

    pqc_prefixes = ("Kyber", "ML-KEM", "NTRU", "Saber", "LightSaber", "FireSaber",
                    "Frodo", "McEliece", "Classic-McEliece", "HQC", "BIKE",
                    "Dilithium", "ML-DSA", "Falcon", "SPHINCS")
    return any(algorithm.startswith(p) for p in pqc_prefixes)


def generate_pqc_key(algorithm: str) -> Tuple[str, str, str]:
    if OQS_AVAILABLE:
        resolved = _resolve_oqs_name(algorithm)
        if resolved:
            oqs_name, category = resolved
            try:
                if category == "kem":
                    raw_key = _generate_via_oqs_kem(oqs_name)
                else:
                    raw_key = _generate_via_oqs_sig(oqs_name)
                return oqs_name, base64.b64encode(raw_key).decode(), "pqc_oqs"
            except Exception as exc:
                logger.warning("OQS generation failed for %s: %s", oqs_name, exc)

    if is_pqc_algorithm(algorithm):
        canonical = _find_canonical_name(algorithm)
        raw_key = _simulate_pqc_key(canonical or algorithm)
        return canonical or algorithm, base64.b64encode(raw_key).decode(), "pqc_simulated"

    raise ValueError(f"PQC algorithm not recognized: {algorithm}")


def _find_canonical_name(requested: str) -> Optional[str]:
    if requested in PQC_KEM_ALGORITHMS:
        return requested
    if requested in PQC_SIGNATURE_ALGORITHMS:
        return requested

    for canonical, meta in PQC_KEM_ALGORITHMS.items():
        if requested in meta["aliases"] or requested == meta["oqs_name"]:
            return canonical
    for canonical, meta in PQC_SIGNATURE_ALGORITHMS.items():
        if requested in meta["aliases"] or requested == meta["oqs_name"]:
            return canonical

    return None


def list_available_pqc_algorithms() -> Dict:
    available_kems = _get_oqs_kem_mechanisms()
    available_sigs = _get_oqs_sig_mechanisms()

    result = {
        "oqs_available": OQS_AVAILABLE,
        "pqcrypto_available": PQC_AVAILABLE,
        "kem_algorithms": {},
        "signature_algorithms": {},
    }

    for canonical, meta in PQC_KEM_ALGORITHMS.items():
        is_available = meta["oqs_name"] in available_kems or canonical in available_kems
        result["kem_algorithms"][canonical] = {
            "family": meta["family"],
            "security_level": meta["security_level"],
            "key_size_bits": meta["key_size_bits"],
            "available_in_oqs": is_available,
            "simulation_available": True,
        }

    for canonical, meta in PQC_SIGNATURE_ALGORITHMS.items():
        is_available = meta["oqs_name"] in available_sigs or canonical in available_sigs
        result["signature_algorithms"][canonical] = {
            "family": meta["family"],
            "security_level": meta["security_level"],
            "available_in_oqs": is_available,
            "simulation_available": True,
        }

    return result


def get_pqc_algorithm_info(algorithm: str) -> Optional[Dict]:
    canonical = _find_canonical_name(algorithm)
    if not canonical:
        return None

    if canonical in PQC_KEM_ALGORITHMS:
        meta = PQC_KEM_ALGORITHMS[canonical]
        return {
            "algorithm": canonical,
            "category": "pqc_kem",
            "family": meta["family"],
            "security_level": meta["security_level"],
            "key_size_bits": meta["key_size_bits"],
            "quantum_resistant": True,
            "recommended": True,
        }

    if canonical in PQC_SIGNATURE_ALGORITHMS:
        meta = PQC_SIGNATURE_ALGORITHMS[canonical]
        return {
            "algorithm": canonical,
            "category": "pqc_signature",
            "family": meta["family"],
            "security_level": meta["security_level"],
            "quantum_resistant": True,
            "recommended": True,
        }

    return None
