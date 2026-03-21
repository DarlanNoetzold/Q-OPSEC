import hashlib
import struct
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from quantum_gateway.channel_simulator import (
    QuantumChannelSimulator,
    QuantumChannelParameters,
    ChannelMetrics,
)


@dataclass
class BB84Result:
    raw_key: bytes
    key_length_bits: int
    quantum_bit_error_rate: float
    sifted_key_length: int
    final_key_length: int
    channel_metrics: ChannelMetrics
    protocol_successful: bool
    error_message: Optional[str] = None


class BB84Protocol:

    QBER_SECURITY_THRESHOLD = 0.11

    def __init__(self, channel: Optional[QuantumChannelSimulator] = None,
                 target_key_bits: int = 256,
                 num_pulses: int = 100000):
        self.channel = channel or QuantumChannelSimulator()
        self.target_key_bits = target_key_bits
        self.num_pulses = num_pulses

    def execute(self) -> BB84Result:
        metrics = ChannelMetrics()
        metrics.total_pulses_sent = self.num_pulses

        alice_bits = self.channel.generate_random_bits(self.num_pulses)
        alice_bases = self.channel.generate_random_bases(self.num_pulses)

        detection_mask = self.channel.simulate_photon_transmission(self.num_pulses)
        metrics.photons_detected = int(np.sum(detection_mask))

        bob_bases = self.channel.generate_random_bases(self.num_pulses)

        detected_alice_bits = alice_bits[detection_mask]
        detected_alice_bases = alice_bases[detection_mask]
        detected_bob_bases = bob_bases[detection_mask]

        matching_basis_mask = detected_alice_bases == detected_bob_bases
        sifted_alice_bits = detected_alice_bits[matching_basis_mask]
        bob_measured_bits = sifted_alice_bits.copy()
        bob_measured_bits = self.channel.apply_depolarization(bob_measured_bits)
        bob_measured_bits = self.channel.apply_phase_errors(bob_measured_bits)
        bob_measured_bits = self.channel.apply_misalignment_noise(bob_measured_bits)

        metrics.sifted_key_length = len(sifted_alice_bits)

        if metrics.sifted_key_length < self.target_key_bits * 4:
            return BB84Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=1.0,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message="Insufficient sifted key material",
            )

        qber, alice_remaining, bob_remaining = self.channel.estimate_qber(
            sifted_alice_bits, bob_measured_bits, sample_fraction=0.1
        )
        metrics.quantum_bit_error_rate = qber

        if qber >= self.QBER_SECURITY_THRESHOLD:
            return BB84Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=qber,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message=f"QBER {qber:.4f} exceeds security threshold {self.QBER_SECURITY_THRESHOLD}",
            )

        reconciled_key = self._cascade_error_correction(alice_remaining, bob_remaining, qber)

        amplified_key = self._privacy_amplification(reconciled_key, qber, self.target_key_bits)
        metrics.final_key_length = len(amplified_key) * 8

        total_time = self.num_pulses / (self.channel.params.clock_rate_mhz * 1e6)
        metrics.transmission_efficiency = metrics.photons_detected / self.num_pulses
        metrics.raw_key_rate_bits_per_second = metrics.final_key_length / total_time if total_time > 0 else 0

        return BB84Result(
            raw_key=amplified_key,
            key_length_bits=len(amplified_key) * 8,
            quantum_bit_error_rate=qber,
            sifted_key_length=metrics.sifted_key_length,
            final_key_length=metrics.final_key_length,
            channel_metrics=metrics,
            protocol_successful=True,
        )

    def _cascade_error_correction(self, alice_bits: np.ndarray,
                                   bob_bits: np.ndarray, qber: float) -> np.ndarray:
        corrected = bob_bits.copy()
        block_size = max(4, int(0.73 / qber)) if qber > 0 else len(alice_bits)

        for cascade_pass in range(4):
            current_block_size = block_size * (2 ** cascade_pass)
            num_blocks = len(alice_bits) // current_block_size

            for block_idx in range(num_blocks):
                start = block_idx * current_block_size
                end = start + current_block_size
                alice_parity = np.sum(alice_bits[start:end]) % 2
                bob_parity = np.sum(corrected[start:end]) % 2

                if alice_parity != bob_parity:
                    corrected = self._binary_search_correction(
                        alice_bits, corrected, start, end
                    )

        return corrected

    def _binary_search_correction(self, alice_bits: np.ndarray,
                                   bob_bits: np.ndarray,
                                   start: int, end: int) -> np.ndarray:
        corrected = bob_bits.copy()

        if end - start <= 1:
            corrected[start] = alice_bits[start]
            return corrected

        mid = (start + end) // 2
        alice_left_parity = np.sum(alice_bits[start:mid]) % 2
        bob_left_parity = np.sum(corrected[start:mid]) % 2

        if alice_left_parity != bob_left_parity:
            corrected = self._binary_search_correction(alice_bits, corrected, start, mid)
        else:
            corrected = self._binary_search_correction(alice_bits, corrected, mid, end)

        return corrected

    def _privacy_amplification(self, reconciled_bits: np.ndarray,
                                qber: float, target_bits: int) -> bytes:
        bit_string = reconciled_bits.astype(np.uint8)
        raw_bytes = np.packbits(bit_string).tobytes()

        target_bytes = target_bits // 8
        qber_bytes = struct.pack(">d", float(qber))
        hash_input = raw_bytes + qber_bytes

        amplified = b""
        counter = 0
        while len(amplified) < target_bytes:
            hash_data = hash_input + counter.to_bytes(4, byteorder="big")
            amplified += hashlib.sha3_256(hash_data).digest()
            counter += 1

        return amplified[:target_bytes]


def run_bb84_simulation(channel_params: Optional[QuantumChannelParameters] = None,
                        target_key_bits: int = 256,
                        num_pulses: int = 100000) -> BB84Result:
    channel = QuantumChannelSimulator(channel_params) if channel_params else QuantumChannelSimulator()
    protocol = BB84Protocol(channel=channel, target_key_bits=target_key_bits, num_pulses=num_pulses)
    return protocol.execute()
