import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from quantum_gateway.channel_simulator import (
    QuantumChannelSimulator,
    QuantumChannelParameters,
    ChannelMetrics,
)


@dataclass
class MDIQKDResult:
    raw_key: bytes
    key_length_bits: int
    quantum_bit_error_rate: float
    bell_state_measurement_success_rate: float
    sifted_key_length: int
    final_key_length: int
    channel_metrics: ChannelMetrics
    protocol_successful: bool
    error_message: Optional[str] = None


class MDIQKDProtocol:

    QBER_THRESHOLD = 0.11
    BSM_SUCCESS_PROBABILITY = 0.5

    def __init__(self, channel_alice: Optional[QuantumChannelSimulator] = None,
                 channel_bob: Optional[QuantumChannelSimulator] = None,
                 target_key_bits: int = 256,
                 num_pulses: int = 200000):
        default_params = QuantumChannelParameters(fiber_length_km=25.0)
        self.channel_alice = channel_alice or QuantumChannelSimulator(default_params)
        self.channel_bob = channel_bob or QuantumChannelSimulator(default_params)
        self.target_key_bits = target_key_bits
        self.num_pulses = num_pulses
        self.rng = np.random.default_rng()

    def execute(self) -> MDIQKDResult:
        metrics = ChannelMetrics()
        metrics.total_pulses_sent = self.num_pulses

        alice_bits = self.channel_alice.generate_random_bits(self.num_pulses)
        alice_bases = self.channel_alice.generate_random_bases(self.num_pulses)
        bob_bits = self.channel_bob.generate_random_bits(self.num_pulses)
        bob_bases = self.channel_bob.generate_random_bases(self.num_pulses)

        alice_arrives = self.channel_alice.simulate_single_photon_detection(self.num_pulses)
        bob_arrives = self.channel_bob.simulate_single_photon_detection(self.num_pulses)
        both_arrive = alice_arrives & bob_arrives

        bsm_success = self.rng.random(self.num_pulses) < self.BSM_SUCCESS_PROBABILITY
        successful_events = both_arrive & bsm_success
        metrics.photons_detected = int(np.sum(successful_events))

        bsm_results = self.rng.integers(0, 4, size=self.num_pulses)

        alice_bits_success = alice_bits[successful_events]
        alice_bases_success = alice_bases[successful_events]
        bob_bits_success = bob_bits[successful_events]
        bob_bases_success = bob_bases[successful_events]
        bsm_results_success = bsm_results[successful_events]

        matching_basis_mask = alice_bases_success == bob_bases_success

        alice_sifted = alice_bits_success[matching_basis_mask]
        bob_sifted = bob_bits_success[matching_basis_mask]
        bsm_sifted = bsm_results_success[matching_basis_mask]

        bob_corrected = bob_sifted.copy()
        phi_plus_mask = (bsm_sifted == 0) | (bsm_sifted == 3)
        bob_corrected[phi_plus_mask] = alice_sifted[phi_plus_mask]
        psi_mask = (bsm_sifted == 1) | (bsm_sifted == 2)
        bob_corrected[psi_mask] = 1 - alice_sifted[psi_mask]

        bob_corrected = self.channel_alice.apply_depolarization(bob_corrected)
        bob_corrected = self.channel_alice.apply_misalignment_noise(bob_corrected)

        metrics.sifted_key_length = len(alice_sifted)
        bsm_success_rate = metrics.photons_detected / self.num_pulses if self.num_pulses > 0 else 0

        if metrics.sifted_key_length < self.target_key_bits * 4:
            return MDIQKDResult(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=1.0,
                bell_state_measurement_success_rate=bsm_success_rate,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message="Insufficient sifted key material for MDI-QKD",
            )

        qber, alice_remaining, bob_remaining = self.channel_alice.estimate_qber(
            alice_sifted, bob_corrected, sample_fraction=0.1
        )
        metrics.quantum_bit_error_rate = qber

        if qber >= self.QBER_THRESHOLD:
            return MDIQKDResult(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=qber,
                bell_state_measurement_success_rate=bsm_success_rate,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message=f"QBER {qber:.4f} exceeds MDI-QKD threshold",
            )

        final_key = self._privacy_amplification(alice_remaining, qber, self.target_key_bits)
        metrics.final_key_length = len(final_key) * 8

        return MDIQKDResult(
            raw_key=final_key,
            key_length_bits=len(final_key) * 8,
            quantum_bit_error_rate=qber,
            bell_state_measurement_success_rate=bsm_success_rate,
            sifted_key_length=metrics.sifted_key_length,
            final_key_length=metrics.final_key_length,
            channel_metrics=metrics,
            protocol_successful=True,
        )

    def _privacy_amplification(self, key_bits: np.ndarray,
                                qber: float, target_bits: int) -> bytes:
        packed = np.packbits(key_bits.astype(np.uint8)).tobytes()
        target_bytes = target_bits // 8
        amplified = b""
        counter = 0
        while len(amplified) < target_bytes:
            hash_input = packed + counter.to_bytes(4, byteorder="big")
            amplified += hashlib.sha3_256(hash_input).digest()
            counter += 1
        return amplified[:target_bytes]


def run_mdi_qkd_simulation(alice_channel_params: Optional[QuantumChannelParameters] = None,
                            bob_channel_params: Optional[QuantumChannelParameters] = None,
                            target_key_bits: int = 256,
                            num_pulses: int = 200000) -> MDIQKDResult:
    alice_channel = QuantumChannelSimulator(alice_channel_params) if alice_channel_params else None
    bob_channel = QuantumChannelSimulator(bob_channel_params) if bob_channel_params else None
    protocol = MDIQKDProtocol(
        channel_alice=alice_channel,
        channel_bob=bob_channel,
        target_key_bits=target_key_bits,
        num_pulses=num_pulses,
    )
    return protocol.execute()
