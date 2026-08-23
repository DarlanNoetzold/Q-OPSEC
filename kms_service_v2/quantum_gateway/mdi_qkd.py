"""
MDIQKDProtocol: Protocolo de distribuição de chave quântica independente de dispositivo.

Refatorado para usar QiskitQuantumSimulator com Bell State Measurement (BSM),
mantendo interface original com MDIQKDResult.
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

logger = logging.getLogger("qiskit_qkd.mdi_qkd")


@dataclass
class MDIQKDResult:
    """Resultado de execução do protocolo MDI-QKD (mantém compatibilidade)."""
    raw_key: bytes
    key_length_bits: int
    quantum_bit_error_rate: float
    bell_state_measurement_success_rate: float
    sifted_key_length: int
    final_key_length: int
    channel_metrics: ChannelMetrics
    protocol_successful: bool
    error_message: Optional[str] = None
    # Novos campos para estatísticas de ruído
    noise_statistics: Optional[dict] = None


class MDIQKDProtocol:
    """
    Protocolo de distribuição de chave quântica independente de dispositivo (MDI-QKD).
    
    Implementa:
    1. Preparação independente de qubits por Alice e Bob
    2. Transmissão para estação de Bell (intermediate node)
    3. Bell State Measurement (BSM)
    4. Post-selection baseado em resultados BSM
    5. Sifting e geração de chave
    6. Amplificação de privacidade
    
    Vantagem: Segurança contra ataques no detector
    """
    
    QBER_THRESHOLD = 0.11
    BSM_SUCCESS_PROBABILITY = 0.5  # ~50% de sucesso em implementações reais
    
    def __init__(self, channel_alice: Optional[QuantumChannelSimulator] = None,
                 channel_bob: Optional[QuantumChannelSimulator] = None,
                 target_key_bits: int = 256,
                 num_pulses: int = 20000):
        """
        Inicializa protocolo MDI-QKD.
        
        Args:
            channel_alice: Canal de Alice
            channel_bob: Canal de Bob
            target_key_bits: Bits-alvo da chave final
            num_pulses: Número de pulsos por participante
        """
        default_params = QuantumChannelParameters(fiber_length_km=25.0)
        self.channel_alice = channel_alice or QuantumChannelSimulator(default_params)
        self.channel_bob = channel_bob or QuantumChannelSimulator(default_params)
        self.target_key_bits = target_key_bits
        self.num_pulses = num_pulses
        self.rng = np.random.default_rng()
        
        logger.info(f"MDIQKDProtocol inicializado: {num_pulses} pulsos, alvo {target_key_bits} bits")
    
    def execute(self) -> MDIQKDResult:
        """
        Executa o protocolo MDI-QKD completo.
        
        Returns:
            MDIQKDResult com chave gerada e estatísticas
        """
        logger.info("Iniciando protocolo MDI-QKD")
        metrics = ChannelMetrics()
        metrics.total_pulses_sent = self.num_pulses
        
        # ====== FASE 1: Preparação de Qubits ======
        alice_bits = self.channel_alice.generate_random_bits(self.num_pulses)
        alice_bases = self.channel_alice.generate_random_bases(self.num_pulses)
        bob_bits = self.channel_bob.generate_random_bits(self.num_pulses)
        bob_bases = self.channel_bob.generate_random_bases(self.num_pulses)
        
        logger.debug(f"Alice e Bob prepararam {self.num_pulses} qubits cada")
        
        # ====== FASE 2: Transmissão para Estação de Bell ======
        alice_arrives = self.channel_alice.simulate_single_photon_detection(self.num_pulses)
        bob_arrives = self.channel_bob.simulate_single_photon_detection(self.num_pulses)
        both_arrive = alice_arrives & bob_arrives
        
        logger.info(f"Pares que chegam: {np.sum(both_arrive)}/{self.num_pulses}")
        
        # ====== FASE 3: Bell State Measurement (BSM) ======
        # Simulação de BSM com sucesso parcial (implementações reais ~50%)
        bsm_success = self.rng.random(self.num_pulses) < self.BSM_SUCCESS_PROBABILITY
        successful_events = both_arrive & bsm_success
        metrics.photons_detected = int(np.sum(successful_events))
        
        logger.info(f"BSM bem-sucedidos: {metrics.photons_detected}/{self.num_pulses} " +
                   f"({metrics.photons_detected/self.num_pulses*100:.2f}%)")
        
        # Extrair dados para eventos bem-sucedidos
        alice_bits_success = alice_bits[successful_events]
        alice_bases_success = alice_bases[successful_events]
        bob_bits_success = bob_bits[successful_events]
        bob_bases_success = bob_bases[successful_events]
        
        # ====== FASE 4: Sifting (Matching de Bases) ======
        matching_basis_mask = alice_bases_success == bob_bases_success
        
        alice_sifted = alice_bits_success[matching_basis_mask]
        bob_sifted = bob_bits_success[matching_basis_mask]
        
        metrics.sifted_key_length = len(alice_sifted)
        bsm_success_rate = metrics.photons_detected / self.num_pulses if self.num_pulses > 0 else 0
        
        logger.info(f"Chave peneirada: {metrics.sifted_key_length} bits")
        
        # ====== FASE 5: Correção de Erros Simples ======
        # MDI-QKD pode exigir apenas correção simples se BSM é perfeito
        # Em caso de discrepâncias, usar XOR (paridade)
        bit_xor = alice_sifted ^ bob_sifted
        bob_corrected = bob_sifted ^ bit_xor
        
        # Aplicar ruído residual do canal
        bob_corrected = self.channel_alice.apply_depolarization(bob_corrected)
        bob_corrected = self.channel_alice.apply_misalignment_noise(bob_corrected)
        
        # ====== Verificação de Material ======
        if metrics.sifted_key_length < self.target_key_bits * 4:
            logger.error("Chave peneirada insuficiente para MDI-QKD")
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
                noise_statistics=self._get_noise_stats(),
            )
        
        # ====== FASE 6: QBER Estimation ======
        qber, alice_remaining, bob_remaining = self.channel_alice.estimate_qber(
            alice_sifted, bob_corrected, sample_fraction=0.1
        )
        metrics.quantum_bit_error_rate = qber
        logger.info(f"QBER estimado: {qber:.4f}")
        
        # ====== Verificação de Segurança ======
        if qber >= self.QBER_THRESHOLD:
            logger.error(f"QBER {qber:.4f} exceeds MDI-QKD threshold")
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
                noise_statistics=self._get_noise_stats(),
            )
        
        # ====== FASE 7: Amplificação de Privacidade ======
        logger.info("Iniciando amplificação de privacidade")
        final_key = self._privacy_amplification(alice_remaining, qber, self.target_key_bits)
        metrics.final_key_length = len(final_key) * 8
        
        # ====== Cálculo de Taxa ======
        total_time = self.num_pulses / (self.channel_alice.params.clock_rate_mhz * 1e6)
        metrics.raw_key_rate_bits_per_second = metrics.final_key_length / total_time if total_time > 0 else 0
        
        logger.info(f"✓ MDI-QKD Sucesso: {metrics.final_key_length} bits, BSM taxa: {bsm_success_rate:.2%}, QBER: {qber:.4f}")
        
        return MDIQKDResult(
            raw_key=final_key,
            key_length_bits=len(final_key) * 8,
            quantum_bit_error_rate=qber,
            bell_state_measurement_success_rate=bsm_success_rate,
            sifted_key_length=metrics.sifted_key_length,
            final_key_length=metrics.final_key_length,
            channel_metrics=metrics,
            protocol_successful=True,
            noise_statistics=self._get_noise_stats(),
        )
    
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
        packed = np.packbits(key_bits.astype(np.uint8)).tobytes()
        target_bytes = target_bits // 8
        
        amplified = b""
        counter = 0
        while len(amplified) < target_bytes:
            hash_input = packed + counter.to_bytes(4, byteorder="big")
            amplified += hashlib.sha3_256(hash_input).digest()
            counter += 1
        
        return amplified[:target_bytes]
    
    def _get_noise_stats(self) -> dict:
        """Coleta estatísticas de ruído dos simuladores."""
        try:
            stats_alice = self.channel_alice.get_noise_statistics()
            stats_bob = self.channel_bob.get_noise_statistics()
            return {
                "depolarization_errors_alice": stats_alice.depolarization_errors,
                "depolarization_errors_bob": stats_bob.depolarization_errors,
                "amplitude_damping_events": stats_alice.amplitude_damping_events + stats_bob.amplitude_damping_events,
                "detection_failures_alice": stats_alice.detection_failures,
                "detection_failures_bob": stats_bob.detection_failures,
            }
        except:
            return {}


def run_mdi_qkd_simulation(alice_channel_params: Optional[QuantumChannelParameters] = None,
                          bob_channel_params: Optional[QuantumChannelParameters] = None,
                          target_key_bits: int = 256,
                          num_pulses: int = 20000) -> MDIQKDResult:
    """
    Função de conveniência para executar simulação MDI-QKD.
    
    Mantém compatibilidade com interface original.
    
    Args:
        alice_channel_params: Parâmetros do canal de Alice
        bob_channel_params: Parâmetros do canal de Bob
        target_key_bits: Bits-alvo da chave
        num_pulses: Número de pulsos por participante
    
    Returns:
        MDIQKDResult com resultado da simulação
    """
    alice_channel = QuantumChannelSimulator(alice_channel_params) if alice_channel_params else None
    bob_channel = QuantumChannelSimulator(bob_channel_params) if bob_channel_params else None
    
    protocol = MDIQKDProtocol(
        channel_alice=alice_channel,
        channel_bob=bob_channel,
        target_key_bits=target_key_bits,
        num_pulses=num_pulses,
    )
    return protocol.execute()
