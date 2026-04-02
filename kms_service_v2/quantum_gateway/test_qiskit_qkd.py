"""
Script de teste comparativo para todos os protocolos QKD com Qiskit.

Executa BB84, E91 e MDI-QKD com diferentes cenários de ruído,
compara resultados e gera relatório detalhado com visualizações.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

from quantum_gateway.bb84 import run_bb84_simulation, BB84Result
from quantum_gateway.e91 import run_e91_simulation, E91Result
from quantum_gateway.mdi_qkd import run_mdi_qkd_simulation, MDIQKDResult
from quantum_gateway.qiskit_channel_simulator import QuantumChannelParameters
from quantum_gateway.logging_utils import StructuredQKDLogger, QKDVisualization


class QKDTestSuite:
    """Suite de testes para protocolos QKD com Qiskit."""
    
    def __init__(self, output_dir: str = "."):
        """Inicializa suite de testes."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.comparison_data = {}
        
        # Criar diretórios
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
    
    def test_bb84_with_noise(self):
        """Testa BB84 com diferentes níveis de ruído."""
        print("\n" + "="*70)
        print("TESTE 1: BB84 com Diferentes Níveis de Ruído")
        print("="*70)
        
        logger = StructuredQKDLogger("BB84", log_dir=self.output_dir / "logs")
        logger.log_event("START", "Iniciando testes de BB84")
        
        noise_scenarios = {
            "low_noise": QuantumChannelParameters(
                fiber_length_km=20.0,
                depolarization_probability=0.001,
                phase_error_probability=0.0005,
                detector_efficiency=0.95,
            ),
            "medium_noise": QuantumChannelParameters(
                fiber_length_km=50.0,
                depolarization_probability=0.01,
                phase_error_probability=0.005,
                detector_efficiency=0.9,
            ),
            "high_noise": QuantumChannelParameters(
                fiber_length_km=100.0,
                depolarization_probability=0.05,
                phase_error_probability=0.02,
                detector_efficiency=0.8,
            ),
        }
        
        results = {}
        for scenario_name, params in noise_scenarios.items():
            print(f"\n  Executando cenário: {scenario_name}")
            logger.log_event("SCENARIO", f"Iniciando cenário {scenario_name}")
            
            start_time = time.time()
            result: BB84Result = run_bb84_simulation(
                channel_params=params,
                target_key_bits=256,
                num_pulses=10000,
            )
            elapsed = time.time() - start_time
            
            results[scenario_name] = {
                "qber": result.quantum_bit_error_rate,
                "key_length": result.final_key_length,
                "key_length_bits": result.key_length_bits,
                "sifted_length": result.sifted_key_length,
                "transmission_efficiency": result.channel_metrics.transmission_efficiency,
                "successful": result.protocol_successful,
                "error_message": result.error_message,
                "elapsed_time": elapsed,
            }
            
            logger.log_metrics({
                "scenario": scenario_name,
                "qber": result.quantum_bit_error_rate,
                "final_key_bits": result.key_length_bits,
                "sifted_key_length": result.sifted_key_length,
                "transmission_efficiency": result.channel_metrics.transmission_efficiency,
                "protocol_successful": result.protocol_successful,
                "execution_time_s": elapsed,
            })
            
            print(f"    ✓ QBER: {result.quantum_bit_error_rate:.4f}")
            print(f"    ✓ Chave: {result.key_length_bits} bits")
            print(f"    ✓ Sucesso: {result.protocol_successful}")
            print(f"    ✓ Tempo: {elapsed:.2f}s")
        
        logger.save_json_report()
        self.results["BB84"] = results
        self.comparison_data["BB84"] = results["medium_noise"]
        
        return results
    
    def test_e91_with_bell(self):
        """Testa E91 com teste de Bell."""
        print("\n" + "="*70)
        print("TESTE 2: E91 com Teste de Desigualdade de Bell")
        print("="*70)
        
        logger = StructuredQKDLogger("E91", log_dir=self.output_dir / "logs")
        logger.log_event("START", "Iniciando testes de E91")
        
        noise_scenarios = {
            "low_noise": QuantumChannelParameters(
                fiber_length_km=20.0,
                depolarization_probability=0.001,
                detector_efficiency=0.95,
            ),
            "medium_noise": QuantumChannelParameters(
                fiber_length_km=50.0,
                depolarization_probability=0.01,
                detector_efficiency=0.9,
            ),
            "high_noise": QuantumChannelParameters(
                fiber_length_km=100.0,
                depolarization_probability=0.05,
                detector_efficiency=0.8,
            ),
        }
        
        results = {}
        for scenario_name, params in noise_scenarios.items():
            print(f"\n  Executando cenário: {scenario_name}")
            logger.log_event("SCENARIO", f"Iniciando cenário {scenario_name}")
            
            start_time = time.time()
            result: E91Result = run_e91_simulation(
                channel_params=params,
                target_key_bits=256,
                num_pairs=100000,
            )
            elapsed = time.time() - start_time
            
            results[scenario_name] = {
                "qber": result.quantum_bit_error_rate,
                "bell_parameter_s": result.bell_parameter_s,
                "bell_violation": result.bell_violation_detected,
                "key_length": result.final_key_length,
                "key_length_bits": result.key_length_bits,
                "sifted_length": result.sifted_key_length,
                "transmission_efficiency": result.channel_metrics.transmission_efficiency,
                "successful": result.protocol_successful,
                "elapsed_time": elapsed,
            }
            
            logger.log_metrics({
                "scenario": scenario_name,
                "qber": result.quantum_bit_error_rate,
                "bell_parameter_s": result.bell_parameter_s,
                "bell_violation_detected": result.bell_violation_detected,
                "final_key_bits": result.key_length_bits,
                "transmission_efficiency": result.channel_metrics.transmission_efficiency,
                "protocol_successful": result.protocol_successful,
                "execution_time_s": elapsed,
            })
            
            print(f"    ✓ Bell S = {result.bell_parameter_s:.4f} (violação: {result.bell_violation_detected})")
            print(f"    ✓ QBER: {result.quantum_bit_error_rate:.4f}")
            print(f"    ✓ Chave: {result.key_length_bits} bits")
            print(f"    ✓ Sucesso: {result.protocol_successful}")
            print(f"    ✓ Tempo: {elapsed:.2f}s")
        
        logger.save_json_report()
        self.results["E91"] = results
        self.comparison_data["E91"] = results["medium_noise"]
        
        return results
    
    def test_mdi_qkd_with_bsm(self):
        """Testa MDI-QKD com Bell State Measurement."""
        print("\n" + "="*70)
        print("TESTE 3: MDI-QKD com Bell State Measurement (BSM)")
        print("="*70)
        
        logger = StructuredQKDLogger("MDI-QKD", log_dir=self.output_dir / "logs")
        logger.log_event("START", "Iniciando testes de MDI-QKD")
        
        noise_scenarios = {
            "low_noise": QuantumChannelParameters(
                fiber_length_km=25.0,
                depolarization_probability=0.001,
                detector_efficiency=0.95,
            ),
            "medium_noise": QuantumChannelParameters(
                fiber_length_km=50.0,
                depolarization_probability=0.01,
                detector_efficiency=0.9,
            ),
            "high_noise": QuantumChannelParameters(
                fiber_length_km=100.0,
                depolarization_probability=0.05,
                detector_efficiency=0.8,
            ),
        }
        
        results = {}
        for scenario_name, params in noise_scenarios.items():
            print(f"\n  Executando cenário: {scenario_name}")
            logger.log_event("SCENARIO", f"Iniciando cenário {scenario_name}")
            
            start_time = time.time()
            result: MDIQKDResult = run_mdi_qkd_simulation(
                alice_channel_params=params,
                bob_channel_params=params,
                target_key_bits=256,
                num_pulses=20000,
            )
            elapsed = time.time() - start_time
            
            results[scenario_name] = {
                "qber": result.quantum_bit_error_rate,
                "bsm_success_rate": result.bell_state_measurement_success_rate,
                "key_length": result.final_key_length,
                "key_length_bits": result.key_length_bits,
                "sifted_length": result.sifted_key_length,
                "transmission_efficiency": result.channel_metrics.transmission_efficiency,
                "successful": result.protocol_successful,
                "elapsed_time": elapsed,
            }
            
            logger.log_metrics({
                "scenario": scenario_name,
                "qber": result.quantum_bit_error_rate,
                "bsm_success_rate": result.bell_state_measurement_success_rate,
                "final_key_bits": result.key_length_bits,
                "transmission_efficiency": result.channel_metrics.transmission_efficiency,
                "protocol_successful": result.protocol_successful,
                "execution_time_s": elapsed,
            })
            
            print(f"    ✓ BSM taxa: {result.bell_state_measurement_success_rate:.2%}")
            print(f"    ✓ QBER: {result.quantum_bit_error_rate:.4f}")
            print(f"    ✓ Chave: {result.key_length_bits} bits")
            print(f"    ✓ Sucesso: {result.protocol_successful}")
            print(f"    ✓ Tempo: {elapsed:.2f}s")
        
        logger.save_json_report()
        self.results["MDI-QKD"] = results
        self.comparison_data["MDI-QKD"] = results["medium_noise"]
        
        return results
    
    def generate_comparison_report(self):
        """Gera relatório comparativo entre protocolos."""
        print("\n" + "="*70)
        print("RELATÓRIO COMPARATIVO")
        print("="*70)
        
        viz = QKDVisualization(output_dir=self.output_dir / "visualizations")
        
        # Comparação de protocolos
        protocols_summary = {}
        for protocol, data in self.comparison_data.items():
            protocols_summary[protocol] = {
                "qber": data.get("qber", 0),
                "key_length": data.get("key_length_bits", 0),
                "transmission_efficiency": data.get("transmission_efficiency", 0),
                "successful": data.get("successful", False),
            }
        
        # Gerar gráfico comparativo
        try:
            filepath = viz.plot_protocol_comparison(protocols_summary)
            print(f"\n✓ Gráfico comparativo salvo: {filepath}")
        except Exception as e:
            print(f"⚠ Erro ao gerar gráfico comparativo: {e}")
        
        # Sumário textual
        summary = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocols_tested": list(self.results.keys()),
            "summary": protocols_summary,
            "detailed_results": self.results,
        }
        
        summary_file = self.output_dir / "test_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Relatório JSON salvo: {summary_file}")
        
        # Imprimir sumário
        print("\n" + "-"*70)
        print("SUMÁRIO POR PROTOCOLO (Cenário: Medium Noise)")
        print("-"*70)
        
        for protocol, data in protocols_summary.items():
            print(f"\n{protocol}:")
            print(f"  QBER:                    {data['qber']:.4f}")
            print(f"  Comprimento da Chave:    {data['key_length']} bits")
            print(f"  Eficiência Transmissão:  {data['transmission_efficiency']:.2%}")
            print(f"  Sucesso:                 {'✓ SIM' if data['successful'] else '✗ NÃO'}")
        
        return summary


def main():
    """Executa suite de testes completa."""
    print("\n" + "="*70)
    print("QKD PROTOCOL TEST SUITE - QISKIT-BASED")
    print("="*70)
    print("Testando: BB84, E91, MDI-QKD")
    print("Simulador: Qiskit AER com Noise Models")
    print("="*70)
    
    try:
        # Criar suite
        suite = QKDTestSuite(output_dir="./test_results")
        
        # Executar testes
        print("\n📊 Executando testes...")
        suite.test_bb84_with_noise()
        suite.test_e91_with_bell()
        suite.test_mdi_qkd_with_bsm()
        
        # Gerar relatório
        suite.generate_comparison_report()
        
        print("\n" + "="*70)
        print("✓ TESTES CONCLUÍDOS COM SUCESSO")
        print("="*70)
        print(f"\nResultados salvos em: {suite.output_dir.absolute()}")
        print("  - Logs: ./logs/")
        print("  - Visualizações: ./visualizations/")
        print("  - Relatório: test_report.json")
        
    except Exception as e:
        print(f"\n✗ ERRO DURANTE TESTES: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()
