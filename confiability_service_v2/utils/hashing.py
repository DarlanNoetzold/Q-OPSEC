"""
Hashing and fingerprinting utilities for trust DNA generation.
"""
import hashlib
import json
from typing import Any, List


def stable_hash(data: Any) -> str:
    """
    Generate stable hash from any data structure.
    Uses JSON serialization with sorted keys.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def trust_dna(vector: List[float], version: str = "v2") -> str:
    """
    Generate Trust DNA from trust vector.
    Format: T-DNA-{version}:{hash}

    Args:
        vector: List of trust dimension scores
        version: DNA version identifier

    Returns:
        Trust DNA string (e.g., "T-DNA-v2:af91c2e4")
    """
    vector_str = ",".join(f"{v:.4f}" for v in vector)
    hash_val = hashlib.sha256(vector_str.encode()).hexdigest()[:8]
    return f"T-DNA-{version}:{hash_val}"


def payload_fingerprint(payload: dict) -> str:
    """
    Generate structural fingerprint of payload.
    Captures structure, not content.
    """
    structure = {
        "keys": sorted(payload.keys()),
        "types": {k: type(v).__name__ for k, v in payload.items()},
        "depth": _get_depth(payload)
    }
    return stable_hash(structure)


def _get_depth(d: dict, level: int = 0) -> int:
    """Calculate nested depth of dictionary."""
    if not isinstance(d, dict) or not d:
        return level
    return max(_get_depth(v, level + 1) if isinstance(v, dict) else level + 1
               for v in d.values())
