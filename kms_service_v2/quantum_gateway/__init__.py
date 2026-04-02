"""
Qiskit QKD Adapter: Protocolos de Distribuição de Chave Quântica com Qiskit.

Este pacote fornece implementações de protocolos QKD usando Qiskit:
- BB84: Protocolo básico de distribuição de chave quântica
- E91: Protocolo com teste de Bell para detecção de espionagem
- MDI-QKD: Protocolo independente de dispositivo com Bell State Measurement

Características:
- Simulação com modelos de ruído realistas (Qiskit AER)
- Suporte a depolarização, amplitude damping, phase damping
- Logs estruturados e visualizações
- Compatibilidade com interface original
"""

from quantum_gateway.qiskit_adapter import (
    QiskitQuantumSimulator,
    NoiseModelConfig,
    NoiseStatistics,
    create_qiskit_simulator_with_noise_preset,
)

from quantum_gateway.qiskit_channel_simulator import (
    QuantumChannelSimulator,
    QuantumChannelParameters,
    ChannelMetrics,
    PolarizationBasis,
    QubitState,
    create_channel_with_distance,
    create_metropolitan_channel,
    create_long_distance_channel,
)

from quantum_gateway.bb84 import (
    BB84Protocol,
    BB84Result,
    run_bb84_simulation,
)

from quantum_gateway.e91 import (
    E91Protocol,
    E91Result,
    run_e91_simulation,
)

from quantum_gateway.mdi_qkd import (
    MDIQKDProtocol,
    MDIQKDResult,
    run_mdi_qkd_simulation,
)

from quantum_gateway.logging_utils import (
    StructuredQKDLogger,
    QKDVisualization,
)

__version__ = "1.0.0"
__author__ = "Qiskit QKD Team"

__all__ = [
    # Adapter
    "QiskitQuantumSimulator",
    "NoiseModelConfig",
    "NoiseStatistics",
    "create_qiskit_simulator_with_noise_preset",
    
    # Channel
    "QuantumChannelSimulator",
    "QuantumChannelParameters",
    "ChannelMetrics",
    "PolarizationBasis",
    "QubitState",
    "create_channel_with_distance",
    "create_metropolitan_channel",
    "create_long_distance_channel",
    
    # Protocols
    "BB84Protocol",
    "BB84Result",
    "run_bb84_simulation",
    
    "E91Protocol",
    "E91Result",
    "run_e91_simulation",
    
    "MDIQKDProtocol",
    "MDIQKDResult",
    "run_mdi_qkd_simulation",
    
    # Utilities
    "StructuredQKDLogger",
    "QKDVisualization",
]
