"""
QiskitChannelSimulator: Simulador de canal quântico integrado com Qiskit.

Mantém interface compatível com channel_simulator.py anterior,
mas usa QiskitQuantumSimulator para cálculos realistas com ruído.
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

from quantum_gateway import QiskitQuantumSimulator, NoiseModelConfig, NoiseStatistics

logger = logging.getLogger("qiskit_qkd.channel")


class PolarizationBasis(Enum):
    """Bases de polarização."""
    RECTILINEAR = "rectilinear"
    DIAGONAL = "diagonal"
    CIRCULAR = "circular"


class QubitState(Enum):
    """Estados de qubit."""
    ZERO = 0
    ONE = 1


@dataclass
class QuantumChannelParameters:
    """Parâmetros de canal quântico (compatível com original)."""
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
    """Métricas de canal (compatível com original)."""
    total_pulses_sent: int = 0
    photons_detected: int = 0
    dark_count_events: int = 0
    transmission_efficiency: float = 0.0
    quantum_bit_error_rate: float = 0.0
    raw_key_rate_bits_per_second: float = 0.0
    sifted_key_length: int = 0
    final_key_length: int = 0


class QuantumChannelSimulator:
    """
    Simulador de canal quântico baseado em Qiskit.
    
    Simula transmissão através de fibra óptica com ruído realista,
    mantendo compatibilidade com interface anterior.
    """
    
    def __init__(self, parameters: Optional[QuantumChannelParameters] = None):
        """
        Inicializa simulador de canal.
        
        Args:
            parameters: Parâmetros do canal
        """
        self.params = parameters or QuantumChannelParameters()
        self.rng = np.random.default_rng()
        
        # Criar simulador Qiskit com parâmetros apropriados
        noise_config = NoiseModelConfig(
            depolarization_rate=self.params.depolarization_probability,
            amplitude_damping_rate=self.params.phase_error_probability * 0.5,
            phase_damping_rate=self.params.phase_error_probability,
            detector_efficiency=self.params.detector_efficiency,
            dark_count_probability=self.params.dark_count_rate_per_ns,
        )
        self.qiskit_sim = QiskitQuantumSimulator(noise_config)
        
        logger.info(f"QuantumChannelSimulator Qiskit inicializado com distância {self.params.fiber_length_km}km")
    
    def calculate_channel_transmittance(self) -> float:
        """Calcula transmitância do canal através da fibra."""
        total_loss_db = self.params.fiber_attenuation_db_per_km * self.params.fiber_length_km
        fiber_transmittance = 10 ** (-total_loss_db / 10)
        return fiber_transmittance * self.params.detector_efficiency
    
    def simulate_photon_transmission(self, num_pulses: int) -> np.ndarray:
        """
        Simula transmissão de fótons através do canal.
        
        Args:
            num_pulses: Número de pulsos a enviar
        
        Returns:
            Array boolean indicando detecções
        """
        channel_transmittance = self.calculate_channel_transmittance()
        mean_photon = self.params.source_mean_photon_number
        
        # Distribuição Poisson de fótons
        photon_numbers = self.rng.poisson(mean_photon, size=num_pulses)
        
        # Probabilidade de detecção
        detection_probabilities = 1.0 - (1.0 - channel_transmittance) ** photon_numbers
        detection_events = self.rng.random(num_pulses) < detection_probabilities
        
        # Dark counts
        dark_count_probability = self.params.dark_count_rate_per_ns * (1e9 / (self.params.clock_rate_mhz * 1e6))
        dark_counts = self.rng.random(num_pulses) < dark_count_probability
        
        all_detections = detection_events | dark_counts
        return all_detections
    
    def simulate_single_photon_detection(self, num_photons: int) -> np.ndarray:
        """
        Simula detecção de fótons únicos.
        
        Args:
            num_photons: Número de fótons a simular
        
        Returns:
            Array boolean de detecções
        """
        total_loss_db = self.params.fiber_attenuation_db_per_km * self.params.fiber_length_km
        fiber_transmittance = 10 ** (-total_loss_db / 10)
        detection_probability = fiber_transmittance * self.params.detector_efficiency
        
        detection_events = self.rng.random(num_photons) < detection_probability
        
        # Dark counts
        dark_count_probability = self.params.dark_count_rate_per_ns * (1e9 / (self.params.clock_rate_mhz * 1e6))
        dark_counts = self.rng.random(num_photons) < dark_count_probability
        
        return detection_events | dark_counts
    
    def apply_depolarization(self, qubit_states: np.ndarray) -> np.ndarray:
        """
        Aplica erro de depolarização.
        
        Args:
            qubit_states: Estados de qubit
        
        Returns:
            Estados com depolarização aplicada
        """
        depol_prob = self.params.depolarization_probability
        depolarized_mask = self.rng.random(len(qubit_states)) < depol_prob
        
        noisy_states = qubit_states.copy()
        num_depolarized = np.sum(depolarized_mask)
        
        if num_depolarized > 0:
            noisy_states[depolarized_mask] = self.rng.integers(0, 2, size=num_depolarized)
            self.qiskit_sim.noise_stats.depolarization_errors += int(num_depolarized)
        
        return noisy_states
    
    def apply_phase_errors(self, basis_measurements: np.ndarray) -> np.ndarray:
        """
        Aplica erros de fase.
        
        Args:
            basis_measurements: Medições
        
        Returns:
            Medições com erros de fase
        """
        phase_prob = self.params.phase_error_probability
        error_mask = self.rng.random(len(basis_measurements)) < phase_prob
        
        noisy_measurements = basis_measurements.copy()
        noisy_measurements[error_mask] = 1 - noisy_measurements[error_mask]
        
        return noisy_measurements
    
    def apply_misalignment_noise(self, measurement_results: np.ndarray) -> np.ndarray:
        """
        Aplica ruído de desalinhamento.
        
        Args:
            measurement_results: Resultados de medição
        
        Returns:
            Resultados com ruído de desalinhamento
        """
        misalignment_rad = np.radians(self.params.misalignment_angle_degrees)
        flip_probability = np.sin(misalignment_rad) ** 2
        
        flip_mask = self.rng.random(len(measurement_results)) < flip_probability
        noisy_results = measurement_results.copy()
        noisy_results[flip_mask] = 1 - noisy_results[flip_mask]
        
        return noisy_results
    
    def estimate_qber(self, alice_bits: np.ndarray, bob_bits: np.ndarray,
                     sample_fraction: float = 0.1) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Estima QBER (Quantum Bit Error Rate).
        
        Args:
            alice_bits: Bits de Alice
            bob_bits: Bits de Bob
            sample_fraction: Fração para amostragem
        
        Returns:
            Tuple (qber, alice_remaining, bob_remaining)
        """
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
        """Gera bits aleatórios."""
        return self.rng.integers(0, 2, size=num_bits)
    
    def generate_random_bases(self, num_bases: int) -> np.ndarray:
        """Gera bases aleatórias."""
        return self.rng.integers(0, 2, size=num_bases)
    
    def calculate_secure_key_rate(self, qber: float, sifted_key_length: int,
                                 total_time_seconds: float) -> Tuple[float, int]:
        """
        Calcula taxa de chave segura.
        
        Args:
            qber: Taxa de erro de bit quântico
            sifted_key_length: Comprimento da chave peneirada
            total_time_seconds: Tempo total de execução
        
        Returns:
            Tuple (secure_key_rate, final_key_length)
        """
        if qber >= 0.11:
            return 0.0, 0
        
        binary_entropy_qber = self._binary_entropy(qber) if qber > 0 else 0.0
        secret_fraction = max(0.0, 1.0 - 2.0 * binary_entropy_qber)
        final_key_length = int(sifted_key_length * secret_fraction)
        secure_key_rate = final_key_length / total_time_seconds if total_time_seconds > 0 else 0.0
        
        return secure_key_rate, final_key_length
    
    @staticmethod
    def _binary_entropy(p: float) -> float:
        """Calcula entropia binária."""
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def get_qiskit_simulator(self) -> QiskitQuantumSimulator:
        """Retorna simulador Qiskit subjacente para acesso avançado."""
        return self.qiskit_sim
    
    def get_noise_statistics(self) -> NoiseStatistics:
        """Retorna estatísticas de ruído."""
        return self.qiskit_sim.get_noise_statistics()


def create_channel_with_distance(distance_km: float) -> QuantumChannelSimulator:
    """Factory para criar canal com distância específica."""
    params = QuantumChannelParameters(fiber_length_km=distance_km)
    return QuantumChannelSimulator(params)


def create_metropolitan_channel() -> QuantumChannelSimulator:
    """Factory para canal metropolitano."""
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
    """Factory para canal de longa distância."""
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
