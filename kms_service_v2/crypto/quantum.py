import base64
import logging
import os
import numpy as np
from typing import Optional, Tuple, Dict

from quantum_gateway.netsquid_adapter import (
    NetSquidBB84Adapter,
    NetSquidE91Adapter,
    NetSquidMDIAdapter,
    get_netsquid_status,
)
from quantum_gateway.channel_simulator import QuantumChannelParameters

logger = logging.getLogger("qopsec.quantum")


def _sanitize_numpy(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_numpy(item) for item in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


QKD_PROTOCOL_MAP = {
    "QKD_BB84": "BB84",
    "QKD_E91": "E91",
    "QKD_MDI": "MDI-QKD",
    "QKD_CV": "BB84",
    "QKD_SARG04": "BB84",
    "QKD_DecoyState": "BB84",
    "QKD_DI": "E91",
}


def is_qkd_enabled() -> bool:
    return os.getenv("QKD_AVAILABLE", "true").lower() == "true"


def is_qkd_algorithm(algorithm: str) -> bool:
    return algorithm.upper().startswith("QKD")


def _estimate_required_pulses(fiber_length_km: float, target_bits: int,
                               detector_efficiency: float = 0.15,
                               mean_photon_number: float = 0.1) -> int:
    attenuation_db = 0.2 * fiber_length_km
    transmittance = 10 ** (-attenuation_db / 10) * detector_efficiency
    detection_rate = mean_photon_number * transmittance
    sifted_rate = detection_rate * 0.5
    safety_factor = 8
    required = int((target_bits * safety_factor) / max(sifted_rate, 1e-10))
    return max(100_000, min(required, 50_000_000))


def generate_qkd_key(algorithm: str,
                      target_bits: int = 256,
                      fiber_length_km: float = 20.0) -> Tuple[str, Optional[str], Dict]:
    if not is_qkd_enabled():
        return algorithm, None, {"error": "QKD gateway disabled via environment"}

    protocol = QKD_PROTOCOL_MAP.get(algorithm, "BB84")

    params = QuantumChannelParameters(
        fiber_length_km=fiber_length_km,
        detector_efficiency=0.15,
        source_mean_photon_number=0.1,
    )
    estimated_pulses = _estimate_required_pulses(fiber_length_km, target_bits)

    if protocol == "BB84":
        from quantum_gateway.bb84 import run_bb84_simulation
        result = run_bb84_simulation(
            channel_params=params,
            target_key_bits=target_bits,
            num_pulses=estimated_pulses,
        )

        if result.protocol_successful and result.raw_key:
            key_b64 = base64.b64encode(result.raw_key).decode()
            metadata = _sanitize_numpy({
                "protocol": protocol,
                "qber": result.quantum_bit_error_rate,
                "sifted_key_length": result.sifted_key_length,
                "final_key_length": result.final_key_length,
                "channel_transmittance": result.channel_metrics.transmission_efficiency,
                "pulses_sent": estimated_pulses,
            })
            return algorithm, key_b64, metadata

        return algorithm, None, _sanitize_numpy({
            "error": result.error_message or "BB84 protocol failed",
            "qber": result.quantum_bit_error_rate,
        })

    if protocol == "E91":
        from quantum_gateway.e91 import run_e91_simulation
        e91_params = QuantumChannelParameters(
            fiber_length_km=min(fiber_length_km, 10.0),
            detector_efficiency=0.25,
            source_mean_photon_number=0.1,
            depolarization_probability=0.005,
        )
        e91_pulses = _estimate_required_pulses(
            min(fiber_length_km, 10.0), target_bits,
            detector_efficiency=0.25,
        )
        e91_pairs = max(2_000_000, int(e91_pulses * 40))
        result = run_e91_simulation(
            channel_params=e91_params,
            target_key_bits=target_bits,
            num_pairs=min(e91_pairs, 50_000_000),
        )

        if result.protocol_successful and result.raw_key:
            key_b64 = base64.b64encode(result.raw_key).decode()
            metadata = _sanitize_numpy({
                "protocol": protocol,
                "qber": result.quantum_bit_error_rate,
                "bell_parameter_s": result.bell_parameter_s,
                "bell_violation": result.bell_violation_detected,
                "sifted_key_length": result.sifted_key_length,
                "final_key_length": result.final_key_length,
            })
            return algorithm, key_b64, metadata

        return algorithm, None, _sanitize_numpy({
            "error": result.error_message or "E91 protocol failed",
            "qber": result.quantum_bit_error_rate,
        })

    if protocol == "MDI-QKD":
        from quantum_gateway.mdi_qkd import run_mdi_qkd_simulation
        mdi_half_distance = min(fiber_length_km / 2, 5.0)
        mdi_params = QuantumChannelParameters(
            fiber_length_km=mdi_half_distance,
            detector_efficiency=0.25,
            source_mean_photon_number=0.1,
            depolarization_probability=0.005,
        )
        mdi_pulses = _estimate_required_pulses(
            mdi_half_distance, target_bits,
            detector_efficiency=0.25,
        )
        mdi_total = max(5_000_000, int(mdi_pulses * 100))
        result = run_mdi_qkd_simulation(
            alice_channel_params=mdi_params,
            bob_channel_params=mdi_params,
            target_key_bits=target_bits,
            num_pulses=min(mdi_total, 50_000_000),
        )

        if result.protocol_successful and result.raw_key:
            key_b64 = base64.b64encode(result.raw_key).decode()
            metadata = _sanitize_numpy({
                "protocol": protocol,
                "qber": result.quantum_bit_error_rate,
                "bsm_success_rate": result.bell_state_measurement_success_rate,
                "sifted_key_length": result.sifted_key_length,
                "final_key_length": result.final_key_length,
            })
            return algorithm, key_b64, metadata

        return algorithm, None, _sanitize_numpy({
            "error": result.error_message or "MDI-QKD protocol failed",
            "qber": result.quantum_bit_error_rate,
        })

    return algorithm, None, {"error": f"Unknown QKD protocol mapping for {algorithm}"}


def list_qkd_algorithms() -> Dict:
    return {
        "enabled": is_qkd_enabled(),
        "netsquid_status": get_netsquid_status(),
        "algorithms": {
            "QKD_BB84": {
                "protocol": "BB84",
                "description": "Bennett-Brassard 1984 prepare-and-measure",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
            "QKD_E91": {
                "protocol": "E91",
                "description": "Ekert 1991 entanglement-based with Bell test",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
            "QKD_MDI": {
                "protocol": "MDI-QKD",
                "description": "Measurement-Device-Independent QKD",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
            "QKD_CV": {
                "protocol": "CV-QKD (mapped to BB84 simulation)",
                "description": "Continuous-Variable QKD",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
            "QKD_SARG04": {
                "protocol": "SARG04 (mapped to BB84 simulation)",
                "description": "Scarani-Acin-Ribordy-Gisin 2004",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
            "QKD_DecoyState": {
                "protocol": "Decoy State (mapped to BB84 simulation)",
                "description": "BB84 with decoy states for PNS attack mitigation",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
            "QKD_DI": {
                "protocol": "DI-QKD (mapped to E91 simulation)",
                "description": "Device-Independent QKD based on Bell violation",
                "security_level": "information_theoretic",
                "quantum_resistant": True,
            },
        },
    }


def get_qkd_algorithm_info(algorithm: str) -> Optional[Dict]:
    all_algorithms = list_qkd_algorithms()["algorithms"]
    info = all_algorithms.get(algorithm)
    if not info:
        return None
    return {
        "algorithm": algorithm,
        "category": "qkd",
        **info,
        "recommended": True,
    }
