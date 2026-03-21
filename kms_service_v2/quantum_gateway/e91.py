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
class E91Result:
    raw_key: bytes
    key_length_bits: int
    quantum_bit_error_rate: float
    bell_parameter_s: float
    bell_violation_detected: bool
    sifted_key_length: int
    final_key_length: int
    channel_metrics: ChannelMetrics
    protocol_successful: bool
    error_message: Optional[str] = None


ALICE_MEASUREMENT_ANGLES = [0.0, 0.0, np.pi / 4]
BOB_MEASUREMENT_ANGLES = [0.0, np.pi / 8, 3 * np.pi / 8]


class E91Protocol:

    BELL_CLASSICAL_BOUND = 2.0
    BELL_QUANTUM_MAXIMUM = 2 * np.sqrt(2)
    QBER_THRESHOLD = 0.11

    def __init__(self, channel: Optional[QuantumChannelSimulator] = None,
                 target_key_bits: int = 256,
                 num_entangled_pairs: int = 150000):
        self.channel = channel or QuantumChannelSimulator()
        self.target_key_bits = target_key_bits
        self.num_pairs = num_entangled_pairs

    def execute(self) -> E91Result:
        metrics = ChannelMetrics()
        metrics.total_pulses_sent = self.num_pairs

        alice_basis_choices = self.channel.rng.integers(0, 3, size=self.num_pairs)
        bob_basis_choices = self.channel.rng.integers(0, 3, size=self.num_pairs)

        alice_angles = np.array([ALICE_MEASUREMENT_ANGLES[b] for b in alice_basis_choices])
        bob_angles = np.array([BOB_MEASUREMENT_ANGLES[b] for b in bob_basis_choices])

        alice_outcomes, bob_outcomes = self._simulate_entangled_measurements(
            alice_angles, bob_angles
        )

        alice_detected = self.channel.simulate_single_photon_detection(self.num_pairs)
        bob_detected = self.channel.simulate_single_photon_detection(self.num_pairs)
        coincidence_mask = alice_detected & bob_detected
        metrics.photons_detected = int(np.sum(coincidence_mask))

        alice_basis_detected = alice_basis_choices[coincidence_mask]
        bob_basis_detected = bob_basis_choices[coincidence_mask]
        alice_outcomes_detected = alice_outcomes[coincidence_mask]
        bob_outcomes_detected = bob_outcomes[coincidence_mask]

        key_generation_mask = (
            (alice_basis_detected == 0) & (bob_basis_detected == 0)
        )
        bell_test_mask = ~key_generation_mask

        bell_s = self._compute_bell_parameter(
            alice_basis_detected[bell_test_mask],
            bob_basis_detected[bell_test_mask],
            alice_outcomes_detected[bell_test_mask],
            bob_outcomes_detected[bell_test_mask],
        )

        bell_violation = abs(bell_s) > self.BELL_CLASSICAL_BOUND

        alice_key_bits = alice_outcomes_detected[key_generation_mask]
        bob_key_bits = 1 - bob_outcomes_detected[key_generation_mask]

        bob_key_bits = self.channel.apply_depolarization(bob_key_bits)
        bob_key_bits = self.channel.apply_misalignment_noise(bob_key_bits)

        metrics.sifted_key_length = len(alice_key_bits)

        if metrics.sifted_key_length < self.target_key_bits * 4:
            return E91Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=1.0,
                bell_parameter_s=bell_s,
                bell_violation_detected=bell_violation,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message="Insufficient key material from coincidence events",
            )

        qber, alice_remaining, bob_remaining = self.channel.estimate_qber(
            alice_key_bits, bob_key_bits, sample_fraction=0.1
        )
        metrics.quantum_bit_error_rate = qber

        if not bell_violation:
            return E91Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=qber,
                bell_parameter_s=bell_s,
                bell_violation_detected=False,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message=f"Bell inequality not violated: S={bell_s:.4f}",
            )

        if qber >= self.QBER_THRESHOLD:
            return E91Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=qber,
                bell_parameter_s=bell_s,
                bell_violation_detected=bell_violation,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message=f"QBER {qber:.4f} exceeds threshold",
            )

        final_key = self._privacy_amplification(alice_remaining, qber, self.target_key_bits)
        metrics.final_key_length = len(final_key) * 8

        return E91Result(
            raw_key=final_key,
            key_length_bits=len(final_key) * 8,
            quantum_bit_error_rate=qber,
            bell_parameter_s=bell_s,
            bell_violation_detected=bell_violation,
            sifted_key_length=metrics.sifted_key_length,
            final_key_length=metrics.final_key_length,
            channel_metrics=metrics,
            protocol_successful=True,
        )

    def _simulate_entangled_measurements(self, alice_angles: np.ndarray,
                                          bob_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle_diff = alice_angles - bob_angles
        correlation = -np.cos(2 * angle_diff)
        uniform_samples = self.channel.rng.random(len(alice_angles))
        alice_outcomes = self.channel.rng.integers(0, 2, size=len(alice_angles))
        prob_same = (1 + correlation) / 2.0
        bob_same_as_alice = uniform_samples < prob_same
        bob_outcomes = np.where(bob_same_as_alice, alice_outcomes, 1 - alice_outcomes)
        return alice_outcomes, bob_outcomes

    def _compute_bell_parameter(self, alice_bases: np.ndarray, bob_bases: np.ndarray,
                                 alice_outcomes: np.ndarray, bob_outcomes: np.ndarray) -> float:
        alice_values = 2 * alice_outcomes.astype(float) - 1
        bob_values = 2 * bob_outcomes.astype(float) - 1
        correlations = {}

        for a_basis in range(3):
            for b_basis in range(3):
                mask = (alice_bases == a_basis) & (bob_bases == b_basis)
                if np.sum(mask) > 10:
                    correlations[(a_basis, b_basis)] = np.mean(
                        alice_values[mask] * bob_values[mask]
                    )

        e_a1_b1 = correlations.get((1, 1), 0.0)
        e_a1_b2 = correlations.get((1, 2), 0.0)
        e_a2_b1 = correlations.get((2, 1), 0.0)
        e_a2_b2 = correlations.get((2, 2), 0.0)

        bell_s = abs(e_a1_b1 - e_a1_b2 + e_a2_b1 + e_a2_b2)
        return bell_s

    def _privacy_amplification(self, key_bits: np.ndarray,
                                qber: float, target_bits: int) -> bytes:
        packed_bytes = np.packbits(key_bits.astype(np.uint8)).tobytes()
        target_bytes = target_bits // 8

        amplified = b""
        counter = 0
        while len(amplified) < target_bytes:
            hash_input = packed_bytes + counter.to_bytes(4, byteorder="big")
            amplified += hashlib.sha3_256(hash_input).digest()
            counter += 1

        return amplified[:target_bytes]


def run_e91_simulation(channel_params: Optional[QuantumChannelParameters] = None,
                       target_key_bits: int = 256,
                       num_pairs: int = 150000) -> E91Result:
    channel = QuantumChannelSimulator(channel_params) if channel_params else QuantumChannelSimulator()
    protocol = E91Protocol(channel=channel, target_key_bits=target_key_bits, num_entangled_pairs=num_pairs)
    return protocol.execute()
