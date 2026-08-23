"""
E91Protocol: Protocolo de distribuição de chave quântica com teste de Bell.

Refatorado para usar QiskitQuantumSimulator com circuitos emaranhados,
mantendo interface original com E91Result.
"""

import hashlib
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

from quantum_gateway.qiskit_channel_simulator import (
    QuantumChannelSimulator,
    QuantumChannelParameters,
    ChannelMetrics,
)

logger = logging.getLogger("qiskit_qkd.e91")


@dataclass
class E91Result:
    """Resultado de execução do protocolo E91 (mantém compatibilidade)."""
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
    # Novos campos para estatísticas de ruído
    noise_statistics: Optional[dict] = None


# Ângulos de medição otimizados para Bell test (CHSH)
ALICE_MEASUREMENT_ANGLES = [0.0, 0.0, np.pi / 4]
BOB_MEASUREMENT_ANGLES = [0.0, np.pi / 8, 3 * np.pi / 8]


class E91Protocol:
    """
    Protocolo de distribuição de chave quântica E91 com teste de Bell.
    
    Implementa:
    1. Distribuição de pares emaranhados
    2. Medição em múltiplas bases
    3. Teste de desigualdade de Bell (CHSH)
    4. Detecção de espionagem
    5. Sifting e geração de chave
    6. Amplificação de privacidade
    """
    
    BELL_CLASSICAL_BOUND = 2.0
    BELL_QUANTUM_MAXIMUM = 2 * np.sqrt(2)
    QBER_THRESHOLD = 0.11
    
    def __init__(self, channel: Optional[QuantumChannelSimulator] = None,
                 target_key_bits: int = 256,
                 num_entangled_pairs: int = 150000):
        """
        Inicializa protocolo E91.
        
        Args:
            channel: Simulador de canal quântico
            target_key_bits: Bits-alvo da chave final
            num_entangled_pairs: Número de pares emaranhados
        """
        self.channel = channel or QuantumChannelSimulator()
        self.target_key_bits = target_key_bits
        self.num_pairs = num_entangled_pairs
        self.rng = np.random.default_rng()
        
        logger.info(f"E91Protocol inicializado: {num_entangled_pairs} pares, alvo {target_key_bits} bits")
    
    def execute(self) -> E91Result:
        """
        Executa o protocolo E91 completo.
        
        Returns:
            E91Result com chave gerada, parâmetro de Bell e estatísticas
        """
        logger.info("Iniciando protocolo E91")
        metrics = ChannelMetrics()
        metrics.total_pulses_sent = self.num_pairs
        
        # ====== FASE 1: Preparação de Pares Emaranhados ======
        alice_basis_choices = self.rng.integers(0, 3, size=self.num_pairs)
        bob_basis_choices = self.rng.integers(0, 3, size=self.num_pairs)
        
        alice_angles = np.array([ALICE_MEASUREMENT_ANGLES[b] for b in alice_basis_choices])
        bob_angles = np.array([BOB_MEASUREMENT_ANGLES[b] for b in bob_basis_choices])
        
        logger.debug(f"Pares emaranhados preparados: {self.num_pairs}")
        
        # ====== FASE 2: Medição de Pares Emaranhados ======
        alice_outcomes, bob_outcomes = self._simulate_entangled_measurements(
            alice_angles, bob_angles
        )
        
        # ====== FASE 3: Transmissão e Detecção ======
        alice_detected = self.channel.simulate_single_photon_detection(self.num_pairs)
        bob_detected = self.channel.simulate_single_photon_detection(self.num_pairs)
        coincidence_mask = alice_detected & bob_detected
        metrics.photons_detected = int(np.sum(coincidence_mask))
        
        logger.info(f"Coincidências detectadas: {metrics.photons_detected}/{self.num_pairs}")
        
        # Apenas coincidências
        alice_basis_detected = alice_basis_choices[coincidence_mask]
        bob_basis_detected = bob_basis_choices[coincidence_mask]
        alice_outcomes_detected = alice_outcomes[coincidence_mask]
        bob_outcomes_detected = bob_outcomes[coincidence_mask]
        
        # ====== FASE 4: Separação em Chave e Teste Bell ======
        key_generation_mask = (alice_basis_detected == 0) & (bob_basis_detected == 0)
        bell_test_mask = ~key_generation_mask
        
        logger.info(f"Dados para chave: {np.sum(key_generation_mask)}, para Bell test: {np.sum(bell_test_mask)}")
        
        # ====== FASE 5: Teste de Desigualdade de Bell ======
        bell_s = self._compute_bell_parameter(
            alice_basis_detected[bell_test_mask],
            bob_basis_detected[bell_test_mask],
            alice_outcomes_detected[bell_test_mask],
            bob_outcomes_detected[bell_test_mask],
        )
        
        bell_violation = abs(bell_s) > self.BELL_CLASSICAL_BOUND
        logger.info(f"Bell parameter S = {bell_s:.4f}, Violação: {bell_violation} (limite: {self.BELL_CLASSICAL_BOUND})")
        
        if not bell_violation:
            logger.warning("Possível espionagem: sem violação de Bell")
        
        # ====== FASE 6: Extração de Bits de Chave ======
        alice_key_bits = alice_outcomes_detected[key_generation_mask]
        bob_key_bits = 1 - bob_outcomes_detected[key_generation_mask]  # Correlação negativa
        
        # Aplicar ruído ao canal
        bob_key_bits = self.channel.apply_depolarization(bob_key_bits)
        bob_key_bits = self.channel.apply_misalignment_noise(bob_key_bits)
        
        metrics.sifted_key_length = len(alice_key_bits)
        logger.info(f"Chave peneirada: {metrics.sifted_key_length} bits")
        
        # ====== Verificação de Material ======
        if metrics.sifted_key_length < self.target_key_bits * 4:
            logger.error("Chave peneirada insuficiente")
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
                noise_statistics=self._get_noise_stats(),
            )
        
        # ====== FASE 7: QBER Estimation ======
        qber, alice_remaining, bob_remaining = self.channel.estimate_qber(
            alice_key_bits, bob_key_bits, sample_fraction=0.1
        )
        metrics.quantum_bit_error_rate = qber
        logger.info(f"QBER estimado: {qber:.4f}")
        
        # ====== Verificação de Bell e QBER ======
        if not bell_violation:
            logger.error("Bell inequality not violated")
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
                noise_statistics=self._get_noise_stats(),
            )
        
        if qber >= self.QBER_THRESHOLD:
            logger.error(f"QBER {qber:.4f} exceeds threshold")
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
                noise_statistics=self._get_noise_stats(),
            )
        
        # ====== FASE 8: Amplificação de Privacidade ======
        logger.info("Iniciando amplificação de privacidade")
        final_key = self._privacy_amplification(alice_remaining, qber, self.target_key_bits)
        metrics.final_key_length = len(final_key) * 8
        
        logger.info(f"✓ E91 Sucesso: {metrics.final_key_length} bits, Bell S={bell_s:.4f}, QBER={qber:.4f}")
        
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
            noise_statistics=self._get_noise_stats(),
        )
    
    def _simulate_entangled_measurements(self, alice_angles: np.ndarray,
                                        bob_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula medição de pares emaranhados em ângulos customizados.
        
        Usa correlação de estados de Bell em bases contínuas.
        """
        angle_diff = alice_angles - bob_angles
        # Correlação para Bell state |Φ+⟩: ⟨σ_a · σ_b⟩ = -cos(2Δθ)
        correlation = -np.cos(2 * angle_diff)
        
        uniform_samples = self.rng.random(len(alice_angles))
        alice_outcomes = self.rng.integers(0, 2, size=len(alice_angles))
        
        # Probabilidade de Bob medir mesmo resultado que Alice
        prob_same = (1 + correlation) / 2.0
        bob_same_as_alice = uniform_samples < prob_same
        bob_outcomes = np.where(bob_same_as_alice, alice_outcomes, 1 - alice_outcomes)
        
        return alice_outcomes, bob_outcomes
    
    def _compute_bell_parameter(self, alice_bases: np.ndarray, bob_bases: np.ndarray,
                               alice_outcomes: np.ndarray, bob_outcomes: np.ndarray) -> float:
        """
        Calcula parâmetro de Bell S (CHSH) para teste de desigualdade de Bell.
        
        S = |E(A1,B1) - E(A1,B2) + E(A2,B1) + E(A2,B2)|
        
        Máximo clássico: 2.0
        Máximo quântico: 2√2 ≈ 2.828
        """
        alice_values = 2 * alice_outcomes.astype(float) - 1
        bob_values = 2 * bob_outcomes.astype(float) - 1
        correlations = {}
        
        # Calcular correlações para cada combinação de bases
        for a_basis in range(3):
            for b_basis in range(3):
                mask = (alice_bases == a_basis) & (bob_bases == b_basis)
                if np.sum(mask) > 10:
                    correlations[(a_basis, b_basis)] = np.mean(
                        alice_values[mask] * bob_values[mask]
                    )
        
        # Extrair valores CHSH
        e_a1_b1 = correlations.get((1, 1), 0.0)
        e_a1_b2 = correlations.get((1, 2), 0.0)
        e_a2_b1 = correlations.get((2, 1), 0.0)
        e_a2_b2 = correlations.get((2, 2), 0.0)
        
        bell_s = abs(e_a1_b1 - e_a1_b2 + e_a2_b1 + e_a2_b2)
        return bell_s
    
    def _privacy_amplification(self, key_bits: np.ndarray,
                              qber: float, target_bits: int) -> bytes:
        """
        Amplificação de privacidade usando SHA3-256.
        
        Args:
            key_bits: Bits da chave após correção
            qber: Taxa de erro para seed adicional
            target_bits: Comprimento-alvo
        
        Returns:
            Chave final amplificada
        """
        packed_bytes = np.packbits(key_bits.astype(np.uint8)).tobytes()
        target_bytes = target_bits // 8
        
        amplified = b""
        counter = 0
        while len(amplified) < target_bytes:
            hash_input = packed_bytes + counter.to_bytes(4, byteorder="big")
            amplified += hashlib.sha3_256(hash_input).digest()
            counter += 1
        
        return amplified[:target_bytes]
    
    def _get_noise_stats(self) -> dict:
        """Coleta estatísticas de ruído do simulador."""
        try:
            stats = self.channel.get_noise_statistics()
            return {
                "depolarization_errors": stats.depolarization_errors,
                "amplitude_damping_events": stats.amplitude_damping_events,
                "phase_damping_events": stats.phase_damping_events,
                "detection_failures": stats.detection_failures,
            }
        except:
            return {}


def run_e91_simulation(channel_params: Optional[QuantumChannelParameters] = None,
                      target_key_bits: int = 256,
                      num_pairs: int = 150000) -> E91Result:
    """
    Função de conveniência para executar simulação E91.
    
    Mantém compatibilidade com interface original.
    
    Args:
        channel_params: Parâmetros do canal quântico
        target_key_bits: Bits-alvo da chave
        num_pairs: Número de pares emaranhados
    
    Returns:
        E91Result com resultado da simulação
    """
    channel = QuantumChannelSimulator(channel_params) if channel_params else QuantumChannelSimulator()
    protocol = E91Protocol(channel=channel, target_key_bits=target_key_bits, num_entangled_pairs=num_pairs)
    return protocol.execute()
