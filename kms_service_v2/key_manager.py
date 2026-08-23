import time
import uuid
import logging
from typing import Optional, Tuple, Dict, Any

from crypto.classical import is_classical_algorithm, generate_classical_key
from crypto.pqc import is_pqc_algorithm, generate_pqc_key, OQS_AVAILABLE, PQC_AVAILABLE
from crypto.quantum import is_qkd_algorithm, generate_qkd_key, is_qkd_enabled
from algorithm_registry import (
    get_fallback_algorithm,
    classify_algorithm,
    get_all_supported_algorithms,
    get_algorithm_info as registry_get_algorithm_info,
)

logger = logging.getLogger("qopsec.key_manager")


def generate_key(algorithm: str) -> Tuple[str, str, str, Optional[Dict]]:
    if is_qkd_algorithm(algorithm):
        selected, key_material, metadata = generate_qkd_key(algorithm)
        if key_material:
            return selected, key_material, "qkd", metadata

        logger.warning("QKD generation failed for %s, applying fallback", algorithm)
        fallback = get_fallback_algorithm(algorithm)
        return _generate_with_category(fallback, "qkd_fallback")

    if is_pqc_algorithm(algorithm):
        try:
            selected, key_material, source = generate_pqc_key(algorithm)
            return selected, key_material, source, None
        except ValueError:
            logger.warning("PQC generation failed for %s, applying fallback", algorithm)
            fallback = get_fallback_algorithm(algorithm)
            return _generate_with_category(fallback, "pqc_fallback")

    if is_classical_algorithm(algorithm):
        key_material = generate_classical_key(algorithm)
        return algorithm, key_material, "classical", None

    raise ValueError(f"Algorithm '{algorithm}' is not supported by the KMS")


def _generate_with_category(algorithm: str, fallback_source: str) -> Tuple[str, str, str, Optional[Dict]]:
    if is_pqc_algorithm(algorithm):
        try:
            selected, key_material, source = generate_pqc_key(algorithm)
            return selected, key_material, source, None
        except ValueError:
            pass

    if is_classical_algorithm(algorithm):
        key_material = generate_classical_key(algorithm)
        return algorithm, key_material, "classical", None

    key_material = generate_classical_key("AES256_GCM")
    return "AES256_GCM", key_material, "classical", None


def build_session(session_id: Optional[str], request_id: Optional[str],
                  algorithm: str, ttl_seconds: int) -> Tuple[str, str, str, str, int, bool, Optional[str], str, Optional[Dict]]:
    selected_algorithm, key_material, source, qkd_metadata = generate_key(algorithm)

    fallback_applied = selected_algorithm != algorithm
    fallback_reason = None

    if fallback_applied:
        category = classify_algorithm(algorithm)
        reason_map = {
            "qkd": "QKD_UNAVAILABLE",
            "pqc": "PQC_LIBRARY_UNAVAILABLE",
            "classical": "ALGORITHM_NOT_SUPPORTED",
            "unknown": "ALGORITHM_NOT_RECOGNIZED",
        }
        fallback_reason = reason_map.get(category, "FALLBACK_APPLIED")

    expires_at = int(time.time()) + max(1, int(ttl_seconds))
    resolved_session_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
    resolved_request_id = request_id or f"req_{uuid.uuid4().hex[:16]}"

    return (
        resolved_session_id,
        resolved_request_id,
        selected_algorithm,
        key_material,
        expires_at,
        fallback_applied,
        fallback_reason,
        source,
        qkd_metadata,
    )


def get_supported_algorithms() -> Dict[str, Any]:
    return get_all_supported_algorithms()


def get_algorithm_info(algorithm: str) -> Optional[Dict[str, Any]]:
    return registry_get_algorithm_info(algorithm)
