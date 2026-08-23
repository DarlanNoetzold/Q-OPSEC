import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional


class PolarizationBasis(Enum):
    RECTILINEAR = "rectilinear"
    DIAGONAL = "diagonal"
    CIRCULAR = "circular"


class QubitState(Enum):
    ZERO = 0
    ONE = 1


@dataclass
class QuantumChannelParameters:
    fiber_length_km: float = 50.0
    fiber_attenuation_db_per_km: float = 0.2
    detector_efficiency: float = 0.1
    dark_count_rate_per_ns: float = 1e-7
    clock_rate_mhz: float = 1000.0
    depolarization_probability: float = 0.01
    phase_error_probability: float = 0.005
    misalignment_angle_degrees: float = 1.5
    source_mean_photon_number: float = 0.1


@dataclass
class ChannelMetrics:
    total_pulses_sent: int = 0
    photons_detected: int = 0
    dark_count_events: int = 0
    transmission_efficiency: float = 0.0
    quantum_bit_error_rate: float = 0.0
    raw_key_rate_bits_per_second: float = 0.0
    sifted_key_length: int = 0
    final_key_length: int = 0


class QuantumChannelSimulator:

    def __init__(self, parameters: Optional[QuantumChannelParameters] = None):
        self.params = parameters or QuantumChannelParameters()
        self.rng = np.random.default_rng()

    def calculate_channel_transmittance(self) -> float:
        total_loss_db = self.params.fiber_attenuation_db_per_km * self.params.fiber_length_km
        fiber_transmittance = 10 ** (-total_loss_db / 10)
        return fiber_transmittance * self.params.detector_efficiency

    def simulate_photon_transmission(self, num_pulses: int) -> np.ndarray:
        channel_transmittance = self.calculate_channel_transmittance()
        mean_photon = self.params.source_mean_photon_number
        photon_numbers = self.rng.poisson(mean_photon, size=num_pulses)
        detection_probabilities = 1.0 - (1.0 - channel_transmittance) ** photon_numbers
        detection_events = self.rng.random(num_pulses) < detection_probabilities
        dark_count_probability = self.params.dark_count_rate_per_ns * (1e9 / (self.params.clock_rate_mhz * 1e6))
        dark_counts = self.rng.random(num_pulses) < dark_count_probability
        all_detections = detection_events | dark_counts
        return all_detections

    def simulate_single_photon_detection(self, num_photons: int) -> np.ndarray:
        total_loss_db = self.params.fiber_attenuation_db_per_km * self.params.fiber_length_km
        fiber_transmittance = 10 ** (-total_loss_db / 10)
        detection_probability = fiber_transmittance * self.params.detector_efficiency
        detection_events = self.rng.random(num_photons) < detection_probability
        dark_count_probability = self.params.dark_count_rate_per_ns * (1e9 / (self.params.clock_rate_mhz * 1e6))
        dark_counts = self.rng.random(num_photons) < dark_count_probability
        return detection_events | dark_counts

    def apply_depolarization(self, qubit_states: np.ndarray) -> np.ndarray:
        depol_prob = self.params.depolarization_probability
        depolarized_mask = self.rng.random(len(qubit_states)) < depol_prob
        noisy_states = qubit_states.copy()
        num_depolarized = np.sum(depolarized_mask)
        if num_depolarized > 0:
            noisy_states[depolarized_mask] = self.rng.integers(0, 2, size=num_depolarized)
        return noisy_states

    def apply_phase_errors(self, basis_measurements: np.ndarray) -> np.ndarray:
        phase_prob = self.params.phase_error_probability
        error_mask = self.rng.random(len(basis_measurements)) < phase_prob
        noisy_measurements = basis_measurements.copy()
        noisy_measurements[error_mask] = 1 - noisy_measurements[error_mask]
        return noisy_measurements

    def apply_misalignment_noise(self, measurement_results: np.ndarray) -> np.ndarray:
        misalignment_rad = np.radians(self.params.misalignment_angle_degrees)
        flip_probability = np.sin(misalignment_rad) ** 2
        flip_mask = self.rng.random(len(measurement_results)) < flip_probability
        noisy_results = measurement_results.copy()
        noisy_results[flip_mask] = 1 - noisy_results[flip_mask]
        return noisy_results

    def estimate_qber(self, alice_bits: np.ndarray, bob_bits: np.ndarray,
                      sample_fraction: float = 0.1) -> Tuple[float, np.ndarray, np.ndarray]:
        num_bits = len(alice_bits)
        sample_size = max(1, int(num_bits * sample_fraction))
        sample_indices = self.rng.choice(num_bits, size=sample_size, replace=False)
        remaining_mask = np.ones(num_bits, dtype=bool)
        remaining_mask[sample_indices] = False

        alice_sample = alice_bits[sample_indices]
        bob_sample = bob_bits[sample_indices]
        errors_in_sample = np.sum(alice_sample != bob_sample)
        qber = errors_in_sample / sample_size if sample_size > 0 else 0.0

        return qber, alice_bits[remaining_mask], bob_bits[remaining_mask]

    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        return self.rng.integers(0, 2, size=num_bits)

    def generate_random_bases(self, num_bases: int) -> np.ndarray:
        return self.rng.integers(0, 2, size=num_bases)

    def calculate_secure_key_rate(self, qber: float, sifted_key_length: int,
                                  total_time_seconds: float) -> Tuple[float, int]:
        if qber >= 0.11:
            return 0.0, 0

        binary_entropy_qber = self._binary_entropy(qber) if qber > 0 else 0.0
        secret_fraction = max(0.0, 1.0 - 2.0 * binary_entropy_qber)
        final_key_length = int(sifted_key_length * secret_fraction)
        secure_key_rate = final_key_length / total_time_seconds if total_time_seconds > 0 else 0.0
        return secure_key_rate, final_key_length

    @staticmethod
    def _binary_entropy(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def create_channel_with_distance(distance_km: float) -> QuantumChannelSimulator:
    params = QuantumChannelParameters(fiber_length_km=distance_km)
    return QuantumChannelSimulator(params)


def create_metropolitan_channel() -> QuantumChannelSimulator:
    params = QuantumChannelParameters(
        fiber_length_km=20.0,
        fiber_attenuation_db_per_km=0.2,
        detector_efficiency=0.15,
        dark_count_rate_per_ns=5e-8,
        clock_rate_mhz=1000.0,
        depolarization_probability=0.005,
        source_mean_photon_number=0.1,
    )
    return QuantumChannelSimulator(params)


def create_long_distance_channel() -> QuantumChannelSimulator:
    params = QuantumChannelParameters(
        fiber_length_km=100.0,
        fiber_attenuation_db_per_km=0.2,
        detector_efficiency=0.1,
        dark_count_rate_per_ns=1e-7,
        clock_rate_mhz=500.0,
        depolarization_probability=0.02,
        source_mean_photon_number=0.05,
    )
    return QuantumChannelSimulator(params)
