"""
Quantum Gateway - QKD Key Generation Interface

This module is responsible only for QKD (Quantum Key Distribution) algorithms.
It does not handle classical or PQC algorithms. Those are managed by the KMS core.
"""

import os
import base64
from quantum_gateway.qkd_interface import (
    qkd_bb84_simulation,
    qkd_e91_simulation,
    qkd_cv_simulation,
    qkd_mdi_simulation,
    qkd_sarg04_simulation,
    qkd_decoy_state_simulation,
    qkd_device_independent_simulation,
)

# Map of supported QKD algorithms to their simulation functions
QKD_ALGORITHMS = {
    "QKD_BB84": qkd_bb84_simulation,
    "QKD_E91": qkd_e91_simulation,
    "QKD_CV": qkd_cv_simulation,
    "QKD_MDI": qkd_mdi_simulation,
    "QKD_SARG04": qkd_sarg04_simulation,
    "QKD_DecoyState": qkd_decoy_state_simulation,
    "QKD_DI": qkd_device_independent_simulation,
}


def generate_key_from_gateway(algorithm: str):
    """
    Try to generate a key using the QKD Gateway.

    Returns:
        (algorithm, Base64-encoded key_material) if QKD is available and supported
        (algorithm, None) if QKD is unavailable or unsupported
    """
    qkd_available = os.getenv("QKD_AVAILABLE", "false").lower() == "true"

    if not qkd_available:
        print(f"[QKD Gateway] QKD not available (env QKD_AVAILABLE != true). Ignoring request for {algorithm}.")
        return algorithm, None

    if algorithm not in QKD_ALGORITHMS:
        if algorithm.upper().startswith("QKD"):
            print(f"[QKD Gateway] Requested QKD algorithm not implemented: {algorithm}")
        else:
            print(f"[QKD Gateway] Ignored non-QKD algorithm: {algorithm}")
        return algorithm, None

    try:
        key = QKD_ALGORITHMS[algorithm]()
        return algorithm, base64.b64encode(key).decode()
    except Exception as e:
        print(f"[QKD Gateway] Error running simulation for {algorithm}: {e}")
        return algorithm, None