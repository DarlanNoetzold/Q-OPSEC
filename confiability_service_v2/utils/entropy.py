"""
Entropy and randomness measurement utilities.
"""
import math
from collections import Counter
from typing import Any, Dict


def shannon_entropy(data: str) -> float:
    """
    Calculate Shannon entropy of a string.
    Higher entropy = more random/unpredictable.

    Returns:
        Entropy value (0.0 to ~8.0 for text)
    """
    if not data:
        return 0.0

    counter = Counter(data)
    length = len(data)

    entropy = 0.0
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def payload_entropy(payload: Dict[str, Any]) -> float:
    """
    Calculate average entropy across all string values in payload.
    """
    string_values = _extract_strings(payload)

    if not string_values:
        return 0.0

    entropies = [shannon_entropy(s) for s in string_values]
    return sum(entropies) / len(entropies)


def _extract_strings(obj: Any) -> list:
    """Recursively extract all string values from nested structure."""
    strings = []

    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(_extract_strings(value))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_extract_strings(item))

    return strings


def is_high_entropy(data: str, threshold: float = 4.5) -> bool:
    """
    Check if data has suspiciously high entropy.
    Useful for detecting encoded/encrypted data.
    """
    return shannon_entropy(data) > threshold