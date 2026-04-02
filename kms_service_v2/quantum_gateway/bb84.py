"""
BB84Protocol: Protocolo de distribuição de chave quântica baseado em Qiskit.

Refatorado para usar QiskitQuantumSimulator mantendo interface original
com BB84Result e mesma assinatura de função.
"""

import hashlib
import struct
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

from quantum_gateway.qiskit_channel_simulator import (
    QuantumChannelSimulator,
    QuantumChannelParameters,
    ChannelMetrics,
)

logger = logging.getLogger("qiskit_qkd.bb84")


@dataclass
class BB84Result:
    """Resultado de execução do protocolo BB84 (mantém compatibilidade)."""
    raw_key: bytes
    key_length_bits: int
    quantum_bit_error_rate: float
    sifted_key_length: int
    final_key_length: int
    channel_metrics: ChannelMetrics
    protocol_successful: bool
    error_message: Optional[str] = None
    # Novos campos para estatísticas de ruído
    noise_statistics: Optional[dict] = None


class BB84Protocol:
    """
    Protocolo de distribuição de chave quântica BB84 com Qiskit.
    
    Implementa:
    1. Preparação de qubits em bases rectilinear/diagonal
    2. Transmissão através de canal com ruído
    3. Sifting (peneiramento de bases)
    4. Correção de erros (Cascade)
    5. Amplificação de privacidade (SHA3-256)
    """
    
    QBER_SECURITY_THRESHOLD = 0.11
    
    def __init__(self, channel: Optional[QuantumChannelSimulator] = None,
                 target_key_bits: int = 256,
                 num_pulses: int = 10000):
        """
        Inicializa protocolo BB84.
        
        Args:
            channel: Simulador de canal quântico
            target_key_bits: Bits-alvo da chave final
            num_pulses: Número de pulsos a transmitir
        """
        self.channel = channel or QuantumChannelSimulator()
        self.target_key_bits = target_key_bits
        self.num_pulses = num_pulses
        self.rng = np.random.default_rng()
        
        logger.info(f"BB84Protocol inicializado: {num_pulses} pulsos, alvo {target_key_bits} bits")
    
    def execute(self) -> BB84Result:
        """
        Executa o protocolo BB84 completo.
        
        Returns:
            BB84Result com chave gerada e estatísticas
        """
        logger.info("Iniciando protocolo BB84")
        metrics = ChannelMetrics()
        metrics.total_pulses_sent = self.num_pulses
        
        # ====== FASE 1: Preparação de Qubits ======
        alice_bits = self.channel.generate_random_bits(self.num_pulses)
        alice_bases = self.channel.generate_random_bases(self.num_pulses)
        
        logger.debug(f"Alice preparou {self.num_pulses} qubits")
        
        # ====== FASE 2: Transmissão through Canal ======
        detection_mask = self.channel.simulate_photon_transmission(self.num_pulses)
        metrics.photons_detected = int(np.sum(detection_mask))
        metrics.transmission_efficiency = metrics.photons_detected / self.num_pulses
        
        logger.info(f"Fótons detectados: {metrics.photons_detected}/{self.num_pulses} ({metrics.transmission_efficiency:.2%})")
        
        # ====== FASE 3: Bob mede em bases aleatórias ======
        bob_bases = self.channel.generate_random_bases(self.num_pulses)
        
        # Apenas qubits detectados
        detected_alice_bits = alice_bits[detection_mask]
        detected_alice_bases = alice_bases[detection_mask]
        detected_bob_bases = bob_bases[detection_mask]
        
        # ====== FASE 4: Sifting (Basis matching) ======
        matching_basis_mask = detected_alice_bases == detected_bob_bases
        sifted_alice_bits = detected_alice_bits[matching_basis_mask]
        
        # Bob mede seus qubits com ruído
        bob_measured_bits = sifted_alice_bits.copy()
        bob_measured_bits = self.channel.apply_depolarization(bob_measured_bits)
        bob_measured_bits = self.channel.apply_phase_errors(bob_measured_bits)
        bob_measured_bits = self.channel.apply_misalignment_noise(bob_measured_bits)
        
        metrics.sifted_key_length = len(sifted_alice_bits)
        logger.info(f"Chave peneirada: {metrics.sifted_key_length} bits")
        
        # ====== Verificação de Material ======
        if metrics.sifted_key_length < self.target_key_bits * 4:
            logger.error("Chave peneirada insuficiente")
            return BB84Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=1.0,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message="Insufficient sifted key material",
                noise_statistics=self._get_noise_stats(),
            )
        
        # ====== FASE 5: QBER Estimation ======
        qber, alice_remaining, bob_remaining = self.channel.estimate_qber(
            sifted_alice_bits, bob_measured_bits, sample_fraction=0.1
        )
        metrics.quantum_bit_error_rate = qber
        logger.info(f"QBER estimado: {qber:.4f}")
        
        if qber >= self.QBER_SECURITY_THRESHOLD:
            logger.error(f"QBER {qber:.4f} exceeds threshold {self.QBER_SECURITY_THRESHOLD}")
            return BB84Result(
                raw_key=b"",
                key_length_bits=0,
                quantum_bit_error_rate=qber,
                sifted_key_length=metrics.sifted_key_length,
                final_key_length=0,
                channel_metrics=metrics,
                protocol_successful=False,
                error_message=f"QBER {qber:.4f} exceeds security threshold {self.QBER_SECURITY_THRESHOLD}",
                noise_statistics=self._get_noise_stats(),
            )
        
        # ====== FASE 6: Error Correction (Cascade) ======
        logger.info("Iniciando correção de erros (Cascade)")
        reconciled_key = self._cascade_error_correction(alice_remaining, bob_remaining, qber)
        
        # ====== FASE 7: Privacy Amplification ======
        logger.info("Iniciando amplificação de privacidade")
        amplified_key = self._privacy_amplification(reconciled_key, qber, self.target_key_bits)
        metrics.final_key_length = len(amplified_key) * 8
        
        # ====== Cálculo de Taxa ======
        total_time = self.num_pulses / (self.channel.params.clock_rate_mhz * 1e6)
        metrics.raw_key_rate_bits_per_second = metrics.final_key_length / total_time if total_time > 0 else 0
        
        logger.info(f"✓ BB84 Sucesso: {metrics.final_key_length} bits, Taxa: {metrics.raw_key_rate_bits_per_second:.2f} bits/s")
        
        return BB84Result(
            raw_key=amplified_key,
            key_length_bits=len(amplified_key) * 8,
            quantum_bit_error_rate=qber,
            sifted_key_length=metrics.sifted_key_length,
            final_key_length=metrics.final_key_length,
            channel_metrics=metrics,
            protocol_successful=True,
            noise_statistics=self._get_noise_stats(),
        )
    
    def _cascade_error_correction(self, alice_bits: np.ndarray,
                                 bob_bits: np.ndarray, qber: float) -> np.ndarray:
        """
        Implementa correção de erros Cascade.
        
        Args:
            alice_bits: Bits de referência (Alice)
            bob_bits: Bits a corrigir (Bob)
            qber: Taxa de erro estimada
        
        Returns:
            Bits corrigidos
        """
        corrected = bob_bits.copy()
        block_size = max(4, int(0.73 / qber)) if qber > 0 else len(alice_bits)
        
        logger.debug(f"Cascade: block_size={block_size}, passes=4")
        
        for cascade_pass in range(4):
            current_block_size = block_size * (2 ** cascade_pass)
            num_blocks = len(alice_bits) // current_block_size
            errors_corrected = 0
            
            for block_idx in range(num_blocks):
                start = block_idx * current_block_size
                end = start + current_block_size
                alice_parity = np.sum(alice_bits[start:end]) % 2
                bob_parity = np.sum(corrected[start:end]) % 2
                
                if alice_parity != bob_parity:
                    corrected, block_errors = self._binary_search_correction(
                        alice_bits, corrected, start, end
                    )
                    errors_corrected += block_errors
        
        logger.debug(f"Cascade concluído")
        return corrected
    
    def _binary_search_correction(self, alice_bits: np.ndarray,
                                 bob_bits: np.ndarray,
                                 start: int, end: int) -> Tuple[np.ndarray, int]:
        """
        Busca binária para encontrar e corrigir erro em bloco.
        
        Returns:
            Tuple (corrected_bits, num_errors_corrected)
        """
        corrected = bob_bits.copy()
        errors = 0
        
        if end - start <= 1:
            if corrected[start] != alice_bits[start]:
                corrected[start] = alice_bits[start]
                errors = 1
            return corrected, errors
        
        mid = (start + end) // 2
        alice_left_parity = np.sum(alice_bits[start:mid]) % 2
        bob_left_parity = np.sum(corrected[start:mid]) % 2
        
        if alice_left_parity != bob_left_parity:
            corrected, left_errors = self._binary_search_correction(
                alice_bits, corrected, start, mid
            )
            errors += left_errors
        else:
            corrected, right_errors = self._binary_search_correction(
                alice_bits, corrected, mid, end
            )
            errors += right_errors
        
        return corrected, errors
    
    def _privacy_amplification(self, reconciled_bits: np.ndarray,
                              qber: float, target_bits: int) -> bytes:
        """
        Amplificação de privacidade usando SHA3-256.
        
        Args:
            reconciled_bits: Bits após correção de erros
            qber: QBER para seed adicional
            target_bits: Comprimento-alvo da chave
        
        Returns:
            Chave final amplificada
        """
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
    
    def _get_noise_stats(self) -> dict:
        """Coleta estatísticas de ruído do simulador."""
        try:
            stats = self.channel.get_noise_statistics()
            return {
                "depolarization_errors": stats.depolarization_errors,
                "amplitude_damping_events": stats.amplitude_damping_events,
                "phase_damping_events": stats.phase_damping_events,
                "detection_failures": stats.detection_failures,
                "initial_fidelity": stats.initial_fidelity,
                "final_fidelity": stats.final_fidelity,
            }
        except:
            return {}


def run_bb84_simulation(channel_params: Optional[QuantumChannelParameters] = None,
                       target_key_bits: int = 256,
                       num_pulses: int = 10000) -> BB84Result:
    """
    Função de conveniência para executar simulação BB84.
    
    Mantém compatibilidade com interface original.
    
    Args:
        channel_params: Parâmetros do canal quântico
        target_key_bits: Bits-alvo da chave
        num_pulses: Número de pulsos
    
    Returns:
        BB84Result com resultado da simulação
    """
    channel = QuantumChannelSimulator(channel_params) if channel_params else QuantumChannelSimulator()
    protocol = BB84Protocol(channel=channel, target_key_bits=target_key_bits, num_pulses=num_pulses)
    return protocol.execute()
