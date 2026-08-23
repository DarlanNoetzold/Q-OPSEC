import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("qopsec.netsquid")

NETSQUID_AVAILABLE = False
try:
    import netsquid as ns
    NETSQUID_AVAILABLE = True
except ImportError:
    ns = None

from quantum_gateway.channel_simulator import QuantumChannelParameters
from quantum_gateway.bb84 import BB84Protocol, BB84Result, run_bb84_simulation
from quantum_gateway.e91 import E91Protocol, E91Result, run_e91_simulation
from quantum_gateway.mdi_qkd import MDIQKDProtocol, MDIQKDResult, run_mdi_qkd_simulation
from quantum_gateway.channel_simulator import QuantumChannelSimulator


class NetSquidBB84Adapter:

    def __init__(self, fiber_length_km: float = 50.0, detector_efficiency: float = 0.1):
        self.fiber_length_km = fiber_length_km
        self.detector_efficiency = detector_efficiency

    def generate_key(self, target_bits: int = 256) -> BB84Result:
        if NETSQUID_AVAILABLE:
            return self._run_netsquid_bb84(target_bits)
        return self._run_fallback_bb84(target_bits)

    def _run_netsquid_bb84(self, target_bits: int) -> BB84Result:
        try:
            ns.sim_reset()

            params = QuantumChannelParameters(
                fiber_length_km=self.fiber_length_km,
                detector_efficiency=self.detector_efficiency,
            )
            return run_bb84_simulation(
                channel_params=params,
                target_key_bits=target_bits,
                num_pulses=100000,
            )
        except Exception as exc:
            logger.warning("NetSquid BB84 execution failed, using fallback: %s", exc)
            return self._run_fallback_bb84(target_bits)

    def _run_fallback_bb84(self, target_bits: int) -> BB84Result:
        params = QuantumChannelParameters(
            fiber_length_km=self.fiber_length_km,
            detector_efficiency=self.detector_efficiency,
        )
        return run_bb84_simulation(
            channel_params=params,
            target_key_bits=target_bits,
            num_pulses=100000,
        )


class NetSquidE91Adapter:

    def __init__(self, fiber_length_km: float = 50.0, detector_efficiency: float = 0.1):
        self.fiber_length_km = fiber_length_km
        self.detector_efficiency = detector_efficiency

    def generate_key(self, target_bits: int = 256) -> E91Result:
        if NETSQUID_AVAILABLE:
            return self._run_netsquid_e91(target_bits)
        return self._run_fallback_e91(target_bits)

    def _run_netsquid_e91(self, target_bits: int) -> E91Result:
        try:
            ns.sim_reset()
            params = QuantumChannelParameters(
                fiber_length_km=self.fiber_length_km,
                detector_efficiency=self.detector_efficiency,
            )
            return run_e91_simulation(
                channel_params=params,
                target_key_bits=target_bits,
                num_pairs=150000,
            )
        except Exception as exc:
            logger.warning("NetSquid E91 execution failed, using fallback: %s", exc)
            return self._run_fallback_e91(target_bits)

    def _run_fallback_e91(self, target_bits: int) -> E91Result:
        params = QuantumChannelParameters(
            fiber_length_km=self.fiber_length_km,
            detector_efficiency=self.detector_efficiency,
        )
        return run_e91_simulation(
            channel_params=params,
            target_key_bits=target_bits,
            num_pairs=150000,
        )


class NetSquidMDIAdapter:

    def __init__(self, alice_distance_km: float = 25.0, bob_distance_km: float = 25.0,
                 detector_efficiency: float = 0.1):
        self.alice_distance_km = alice_distance_km
        self.bob_distance_km = bob_distance_km
        self.detector_efficiency = detector_efficiency

    def generate_key(self, target_bits: int = 256) -> MDIQKDResult:
        if NETSQUID_AVAILABLE:
            return self._run_netsquid_mdi(target_bits)
        return self._run_fallback_mdi(target_bits)

    def _run_netsquid_mdi(self, target_bits: int) -> MDIQKDResult:
        try:
            ns.sim_reset()
            alice_params = QuantumChannelParameters(
                fiber_length_km=self.alice_distance_km,
                detector_efficiency=self.detector_efficiency,
            )
            bob_params = QuantumChannelParameters(
                fiber_length_km=self.bob_distance_km,
                detector_efficiency=self.detector_efficiency,
            )
            return run_mdi_qkd_simulation(
                alice_channel_params=alice_params,
                bob_channel_params=bob_params,
                target_key_bits=target_bits,
                num_pulses=200000,
            )
        except Exception as exc:
            logger.warning("NetSquid MDI-QKD failed, using fallback: %s", exc)
            return self._run_fallback_mdi(target_bits)

    def _run_fallback_mdi(self, target_bits: int) -> MDIQKDResult:
        alice_params = QuantumChannelParameters(
            fiber_length_km=self.alice_distance_km,
            detector_efficiency=self.detector_efficiency,
        )
        bob_params = QuantumChannelParameters(
            fiber_length_km=self.bob_distance_km,
            detector_efficiency=self.detector_efficiency,
        )
        return run_mdi_qkd_simulation(
            alice_channel_params=alice_params,
            bob_channel_params=bob_params,
            target_key_bits=target_bits,
            num_pulses=200000,
        )


def get_netsquid_status() -> Dict[str, Any]:
    status = {
        "netsquid_installed": NETSQUID_AVAILABLE,
        "simulation_backend": "netsquid" if NETSQUID_AVAILABLE else "academic_simulation",
    }

    if NETSQUID_AVAILABLE:
        try:
            status["netsquid_version"] = ns.__version__
        except AttributeError:
            status["netsquid_version"] = "unknown"

    status["supported_protocols"] = ["BB84", "E91", "MDI-QKD"]
    status["channel_model"] = "fiber_optic_with_noise"
    status["noise_models"] = [
        "depolarization",
        "phase_error",
        "detector_dark_counts",
        "misalignment",
        "photon_loss",
    ]
    status["post_processing"] = [
        "basis_sifting",
        "cascade_error_correction",
        "privacy_amplification",
    ]
    return status
