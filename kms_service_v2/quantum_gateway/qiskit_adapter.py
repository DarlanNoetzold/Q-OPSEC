"""
QiskitQuantumSimulator: Adaptador Qiskit para simulação de protocolos QKD.

Substitui o adaptador NetSquid com implementação nativa Qiskit,
suportando modelos de ruído realistas: depolarização, amplitude damping, phase damping.
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        pauli_error,
        amplitude_damping_error,
        phase_damping_error,
        depolarizing_error,
    )
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

logger = logging.getLogger("qiskit_qkd.adapter")


@dataclass
class NoiseModelConfig:
    """Configuração de parâmetros de ruído."""
    # Depolarização (erro de bit aleatório)
    depolarization_rate: float = 0.01
    
    # Amplitude damping (dissipação de energia)
    amplitude_damping_rate: float = 0.005
    t1_relaxation_time_us: float = 100.0  # T1 em microsegundos
    
    # Phase damping (perda de coerência)
    phase_damping_rate: float = 0.003
    t2_dephasing_time_us: float = 50.0  # T2 em microsegundos
    
    # Parâmetros de detecção
    detector_efficiency: float = 0.9
    dark_count_probability: float = 1e-6
    
    # Tempo de gate (para calcular dissipação)
    gate_time_ns: float = 40.0


@dataclass
class NoiseStatistics:
    """Estatísticas de impacto de ruído."""
    initial_fidelity: float = 1.0
    final_fidelity: float = 1.0
    fidelity_loss_percent: float = 0.0
    
    depolarization_errors: int = 0
    amplitude_damping_events: int = 0
    phase_damping_events: int = 0
    detection_failures: int = 0
    
    detailed_log: List[str] = field(default_factory=list)


class QiskitQuantumSimulator:
    """
    Simulador quântico baseado em Qiskit com suporte a modelos de ruído realistas.
    
    Implementa preparação de estados, aplicação de gates, medição e
    cálculo de fidelidade com impacto de ruído.
    """
    
    def __init__(self, noise_config: Optional[NoiseModelConfig] = None):
        """
        Inicializa o simulador Qiskit.
        
        Args:
            noise_config: Configuração de parâmetros de ruído
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit não está instalado. Execute: pip install qiskit qiskit-aer")
        
        self.noise_config = noise_config or NoiseModelConfig()
        self.rng = np.random.default_rng()
        self.noise_model = self._build_noise_model()
        self.simulator = AerSimulator(noise_model=self.noise_model)
        self.noise_stats = NoiseStatistics()
        
        logger.info(f"QiskitQuantumSimulator inicializado com config: {self.noise_config}")
    
    def _build_noise_model(self) -> NoiseModel:
        """Constrói o modelo de ruído personalizado."""
        noise_model = NoiseModel()
        
        # Depolarização em single-qubit gates
        depol_error = depolarizing_error(self.noise_config.depolarization_rate, 1)
        noise_model.add_all_qubit_quantum_error(depol_error, ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz'])
        
        # Amplitude damping (dissipação)
        amp_damp = amplitude_damping_error(self.noise_config.amplitude_damping_rate)
        noise_model.add_all_qubit_quantum_error(amp_damp, ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
        
        # Phase damping (dephasing)
        phase_damp = phase_damping_error(self.noise_config.phase_damping_rate)
        noise_model.add_all_qubit_quantum_error(phase_damp, ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
        
        # Two-qubit gates (CNOT) com ruído aumentado
        cnot_depol = depolarizing_error(self.noise_config.depolarization_rate * 2, 2)
        noise_model.add_all_qubit_quantum_error(cnot_depol, ['cx'])
        
        # Readout error (erros de medição)
        readout_error_prob = 1 - self.noise_config.detector_efficiency
        readout_error = np.array(
            [[1 - readout_error_prob, readout_error_prob],
             [readout_error_prob, 1 - readout_error_prob]]
        )
        
        return noise_model
    
    def prepare_qubit_state(self, bit: int, basis: int) -> QuantumCircuit:
        """
        Prepara um qubit em estado |0⟩ ou |1⟩ em base rectilinear ou diagonal.
        
        Args:
            bit: 0 ou 1
            basis: 0 (rectilinear/Z) ou 1 (diagonal/X)
        
        Returns:
            QuantumCircuit com o qubit preparado
        """
        qc = QuantumCircuit(1, 1, name=f"prepare_{bit}_basis_{basis}")
        
        # Prepare |0⟩ ou |1⟩
        if bit == 1:
            qc.x(0)
        
        # Aplicar rotação para a base diagonal
        if basis == 1:  # Diagonal (X basis)
            qc.ry(np.pi / 4, 0)  # Rotação para base diagonal
        
        self.noise_stats.detailed_log.append(f"Estado preparado: |{bit}⟩ em base {basis}")
        return qc
    
    def apply_measurement_basis(self, circuit: QuantumCircuit, basis: int) -> QuantumCircuit:
        """
        Aplica a base de medição.
        
        Args:
            circuit: Circuito a modificar
            basis: 0 (Z-basis rectilinear) ou 1 (X-basis diagonal)
        
        Returns:
            Circuito com base de medição aplicada
        """
        if basis == 1:  # Medir na base X
            circuit.ry(-np.pi / 4, 0)  # Rotação inversa para medir na base X
        
        circuit.measure(0, 0)
        return circuit
    
    def measure_with_noise(self, circuit: QuantumCircuit, shots: int = 1) -> np.ndarray:
        """
        Executa medição com modelo de ruído.
        
        Args:
            circuit: Circuito a executar
            shots: Número de disparos
        
        Returns:
            Array de resultados de medição (0 ou 1)
        """
        # Executar com ruído
        job = self.simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        
        # Converter contagens para array
        measurements = []
        for bitstring, count in counts.items():
            bit = int(bitstring, 2)
            measurements.extend([bit] * count)
        
        measurements = np.array(measurements)
        
        # Aplicar probabilidade de dark count
        if self.noise_config.dark_count_probability > 0:
            dark_count_mask = self.rng.random(len(measurements)) < self.noise_config.dark_count_probability
            measurements[dark_count_mask] = 1 - measurements[dark_count_mask]
            self.noise_stats.detection_failures += np.sum(dark_count_mask)
        
        return measurements
    
    def create_entangled_pair(self) -> QuantumCircuit:
        """
        Cria um par emaranhado Bell (Bell state |Φ+⟩).
        
        Returns:
            QuantumCircuit com par emaranhado
        """
        qc = QuantumCircuit(2, 2, name="bell_pair")
        qc.h(0)
        qc.cx(0, 1)
        
        self.noise_stats.detailed_log.append("Par emaranhado criado (|Φ+⟩)")
        return qc
    
    def perform_bell_state_measurement(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Realiza medição de estado de Bell (BSM - Bell State Measurement).
        
        Args:
            circuit: Circuito contendo par emaranhado
        
        Returns:
            Circuito com BSM aplicado
        """
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.measure([0, 1], [0, 1])
        
        return circuit
    
    def measure_entangled_system(self, alice_angle: float, bob_angle: float) -> Tuple[int, int]:
        """
        Mede um sistema emaranhado em ângulos específicos (E91).
        
        Args:
            alice_angle: Ângulo de medição de Alice
            bob_angle: Ângulo de medição de Bob
        
        Returns:
            Tuple (alice_outcome, bob_outcome)
        """
        qc = QuantumCircuit(2, 2, name="entangled_measurement")
        
        # Criar par emaranhado
        qc.h(0)
        qc.cx(0, 1)
        
        # Aplicar rotações customizadas
        qc.ry(alice_angle, 0)
        qc.ry(bob_angle, 1)
        
        # Medir
        job = self.simulator.run(qc, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        bitstring = list(counts.keys())[0]
        
        alice_outcome = int(bitstring[1])
        bob_outcome = int(bitstring[0])
        
        return alice_outcome, bob_outcome
    
    def calculate_fidelity(self, ideal_state: np.ndarray, noisy_state: np.ndarray) -> float:
        """
        Calcula fidelidade entre estado ideal e ruidoso.
        
        Args:
            ideal_state: Vetor de estado ideal
            noisy_state: Vetor de estado com ruído
        
        Returns:
            Fidelidade (0.0 a 1.0)
        """
        # Normalizar
        ideal_state = ideal_state / np.linalg.norm(ideal_state)
        noisy_state = noisy_state / np.linalg.norm(noisy_state)
        
        # F = |⟨ψ|φ⟩|^2
        overlap = np.abs(np.dot(np.conj(ideal_state), noisy_state)) ** 2
        return float(overlap)
    
    def simulate_bb84_preparation(self, bit: int, basis: int) -> Dict[str, Any]:
        """
        Simula preparação BB84 com medição em base aleatória.
        
        Args:
            bit: Bit a enviar (0 ou 1)
            basis: Base de preparação (0 ou 1)
        
        Returns:
            Dict com resultado da simulação
        """
        circuit = self.prepare_qubit_state(bit, basis)
        
        # Medir em base aleatória (Alice não conhece a base de Bob)
        measurement_basis = self.rng.integers(0, 2)
        circuit = self.apply_measurement_basis(circuit, measurement_basis)
        
        result = self.measure_with_noise(circuit, shots=1)[0]
        
        return {
            "prepared_bit": bit,
            "prepared_basis": basis,
            "measurement_basis": measurement_basis,
            "measured_result": result,
            "basis_match": basis == measurement_basis,
        }
    
    def get_noise_statistics(self) -> NoiseStatistics:
        """Retorna estatísticas atualizadas de ruído."""
        return self.noise_stats
    
    def reset_statistics(self):
        """Reseta estatísticas de ruído."""
        self.noise_stats = NoiseStatistics()
        logger.info("Estatísticas de ruído resetadas")
    
    def generate_random_bits(self, num_bits: int) -> np.ndarray:
        """Gera bits aleatórios."""
        return self.rng.integers(0, 2, size=num_bits)
    
    def generate_random_bases(self, num_bases: int) -> np.ndarray:
        """Gera bases aleatórias."""
        return self.rng.integers(0, 2, size=num_bases)
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do simulador."""
        return {
            "qiskit_available": QISKIT_AVAILABLE,
            "backend": "AerSimulator",
            "noise_model_enabled": True,
            "depolarization_rate": self.noise_config.depolarization_rate,
            "amplitude_damping_rate": self.noise_config.amplitude_damping_rate,
            "phase_damping_rate": self.noise_config.phase_damping_rate,
            "detector_efficiency": self.noise_config.detector_efficiency,
        }


def create_qiskit_simulator_with_noise_preset(preset: str) -> QiskitQuantumSimulator:
    """
    Factory para criar simulador com preset de ruído.
    
    Args:
        preset: 'low_noise', 'medium_noise', 'high_noise', ou 'realistic'
    
    Returns:
        QiskitQuantumSimulator configurado
    """
    presets = {
        "low_noise": NoiseModelConfig(
            depolarization_rate=0.001,
            amplitude_damping_rate=0.0005,
            phase_damping_rate=0.0003,
            detector_efficiency=0.95,
        ),
        "medium_noise": NoiseModelConfig(
            depolarization_rate=0.01,
            amplitude_damping_rate=0.005,
            phase_damping_rate=0.003,
            detector_efficiency=0.90,
        ),
        "high_noise": NoiseModelConfig(
            depolarization_rate=0.05,
            amplitude_damping_rate=0.02,
            phase_damping_rate=0.01,
            detector_efficiency=0.80,
        ),
        "realistic": NoiseModelConfig(
            depolarization_rate=0.02,
            amplitude_damping_rate=0.01,
            phase_damping_rate=0.005,
            detector_efficiency=0.85,
            t1_relaxation_time_us=100.0,
            t2_dephasing_time_us=50.0,
        ),
    }
    
    if preset not in presets:
        raise ValueError(f"Preset unknown: {preset}. Opções: {list(presets.keys())}")
    
    return QiskitQuantumSimulator(presets[preset])
