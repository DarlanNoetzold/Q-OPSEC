import logging
from typing import Dict, Optional, Any, List

from crypto.classical import list_classical_algorithms
from crypto.pqc import list_available_pqc_algorithms, get_pqc_algorithm_info, PQC_KEM_ALGORITHMS, PQC_SIGNATURE_ALGORITHMS
from crypto.quantum import list_qkd_algorithms, get_qkd_algorithm_info
from hardware_profiler import (
    get_hardware_profile,
    get_algorithm_availability,
    suggest_alternative,
    AlgorithmAvailability,
    HardwareProfile,
)

logger = logging.getLogger("qopsec.registry")


FALLBACK_CHAIN = {
    "QKD_BB84": ["Kyber1024", "AES256_GCM"],
    "QKD_E91": ["Kyber1024", "AES256_GCM"],
    "QKD_MDI": ["Kyber1024", "AES256_GCM"],
    "QKD_CV": ["Kyber768", "AES256_GCM"],
    "QKD_SARG04": ["Kyber768", "AES256_GCM"],
    "QKD_DecoyState": ["Kyber768", "AES256_GCM"],
    "QKD_DI": ["Kyber1024", "AES256_GCM"],
    "Kyber1024": ["Kyber768", "AES256_GCM"],
    "Kyber768": ["Kyber512", "AES256_GCM"],
    "Kyber512": ["AES256_GCM"],
    "Classic-McEliece-8192128": ["Kyber1024", "AES256_GCM"],
    "Classic-McEliece-6960119": ["Kyber1024", "AES256_GCM"],
    "Classic-McEliece-6688128": ["Kyber768", "AES256_GCM"],
    "Classic-McEliece-460896": ["Kyber768", "AES256_GCM"],
    "Classic-McEliece-348864": ["Kyber512", "AES256_GCM"],
    "FrodoKEM-1344-AES": ["Kyber1024", "AES256_GCM"],
    "FrodoKEM-976-AES": ["Kyber768", "AES256_GCM"],
    "FrodoKEM-640-AES": ["Kyber512", "AES256_GCM"],
    "HQC-256": ["Kyber1024", "AES256_GCM"],
    "HQC-192": ["Kyber768", "AES256_GCM"],
    "HQC-128": ["Kyber512", "AES256_GCM"],
    "BIKE-L5": ["Kyber1024", "AES256_GCM"],
    "BIKE-L3": ["Kyber768", "AES256_GCM"],
    "BIKE-L1": ["Kyber512", "AES256_GCM"],
    "NTRU-HPS-4096-821": ["Kyber1024", "AES256_GCM"],
    "NTRU-HPS-2048-677": ["Kyber768", "AES256_GCM"],
    "NTRU-HPS-2048-509": ["Kyber512", "AES256_GCM"],
    "NTRU-HRSS-701": ["Kyber768", "AES256_GCM"],
    "LightSaber": ["Kyber512", "AES256_GCM"],
    "Saber": ["Kyber768", "AES256_GCM"],
    "FireSaber": ["Kyber1024", "AES256_GCM"],
    "RSA4096": ["ECDH_P256", "AES256_GCM"],
    "RSA2048": ["ECDH_P256", "AES256_GCM"],
}

DEFAULT_FALLBACK = "AES256_GCM"


def get_fallback_algorithm(algorithm: str) -> str:
    chain = FALLBACK_CHAIN.get(algorithm, [DEFAULT_FALLBACK])
    profile = get_hardware_profile()

    for candidate in chain:
        availability = get_algorithm_availability(candidate, profile)
        if availability == AlgorithmAvailability.AVAILABLE:
            return candidate

    return DEFAULT_FALLBACK


def get_all_supported_algorithms() -> Dict[str, Any]:
    classical = list_classical_algorithms()
    pqc = list_available_pqc_algorithms()
    qkd = list_qkd_algorithms()

    return {
        "classical": list(classical.keys()),
        "pqc_kems": list(pqc["kem_algorithms"].keys()),
        "pqc_signatures": list(pqc["signature_algorithms"].keys()),
        "qkd": list(qkd["algorithms"].keys()),
        "oqs_available": pqc["oqs_available"],
        "pqcrypto_available": pqc["pqcrypto_available"],
        "qkd_enabled": qkd["enabled"],
        "details": {
            "classical": classical,
            "pqc": pqc,
            "qkd": qkd,
        },
    }


def get_algorithm_info(algorithm: str) -> Optional[Dict[str, Any]]:
    classical_algos = list_classical_algorithms()
    if algorithm in classical_algos:
        meta = classical_algos[algorithm]
        return {
            "algorithm": algorithm,
            **meta,
        }

    pqc_info = get_pqc_algorithm_info(algorithm)
    if pqc_info:
        return pqc_info

    qkd_info = get_qkd_algorithm_info(algorithm)
    if qkd_info:
        return qkd_info

    return None


def classify_algorithm(algorithm: str) -> str:
    from crypto.classical import is_classical_algorithm
    from crypto.pqc import is_pqc_algorithm
    from crypto.quantum import is_qkd_algorithm

    if is_qkd_algorithm(algorithm):
        return "qkd"
    if is_pqc_algorithm(algorithm):
        return "pqc"
    if is_classical_algorithm(algorithm):
        return "classical"
    return "unknown"


def get_algorithm_with_availability(algorithm: str) -> Dict[str, Any]:
    profile = get_hardware_profile()
    availability = get_algorithm_availability(algorithm, profile)
    info = get_algorithm_info(algorithm)

    result = {
        "algorithm": algorithm,
        "availability": availability.value,
        "category": classify_algorithm(algorithm),
        "info": info,
    }

    if availability in (AlgorithmAvailability.SLOW, AlgorithmAvailability.UNAVAILABLE):
        alternative = suggest_alternative(algorithm, profile)
        if alternative:
            result["suggested_alternative"] = alternative

    return result
