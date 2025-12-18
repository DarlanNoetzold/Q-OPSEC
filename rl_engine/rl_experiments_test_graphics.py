#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYNTHETIC RL ENGINE EXPERIMENT - REALISTIC VERSION
Gera dados artificiais determinísticos com:
 - Variação realista entre algoritmos (alguns mais usados, outros menos)
 - Taxa de sucesso realista (~85-92%, não 100%)
 - Ruído nos dados (latência, recursos, etc.)
 - Falhas ocasionais
 - Padrões mais naturais
"""

import json
import csv
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any

# ====================================================================== #
# Lista de algoritmos                                                    #
# ====================================================================== #
ALGORITHMS = [
    "QKD_BB84", "QKD_E91", "QKD_CV-QKD", "QKD_MDI-QKD", "QKD_DECOY",
    "PQC_KYBER", "PQC_DILITHIUM", "PQC_NTRU", "PQC_SABER", "PQC_FALCON",
    "PQC_SPHINCS", "HYBRID_QKD_PQC", "HYBRID_RSA_PQC", "HYBRID_ECC_PQC",
    "RSA_4096", "ECC_521", "AES_256_GCM", "AES_192", "CHACHA20_POLY1305",
    "FALLBACK_AES"
]


class SyntheticRLExperiment:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.algorithm_usage_counter: Dict[str, int] = {}

        # Pesos de preferência por algoritmo (alguns mais usados que outros)
        self.algorithm_weights = {
            "PQC_KYBER": 1.8,  # Muito usado
            "PQC_DILITHIUM": 1.6,  # Muito usado
            "HYBRID_QKD_PQC": 1.4,  # Bastante usado
            "QKD_BB84": 1.3,  # Bastante usado
            "AES_256_GCM": 1.2,  # Comum
            "PQC_FALCON": 1.1,  # Comum
            "RSA_4096": 1.0,  # Normal
            "ECC_521": 1.0,  # Normal
            "PQC_NTRU": 0.9,  # Menos usado
            "QKD_E91": 0.8,  # Menos usado
            "HYBRID_RSA_PQC": 0.8,  # Menos usado
            "QKD_CV-QKD": 0.7,  # Pouco usado
            "PQC_SABER": 0.7,  # Pouco usado
            "CHACHA20_POLY1305": 0.6,  # Pouco usado
            "QKD_MDI-QKD": 0.5,  # Raro
            "QKD_DECOY": 0.5,  # Raro
            "HYBRID_ECC_PQC": 0.5,  # Raro
            "PQC_SPHINCS": 0.4,  # Muito raro
            "AES_192": 0.3,  # Muito raro
            "FALLBACK_AES": 0.2  # Emergência apenas
        }

    # ------------------------------------------------------------------ #
    # Utilitários determinísticos com ruído                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _algo_family(algo: str) -> str:
        if "QKD" in algo:
            return "QKD"
        if "PQC" in algo:
            return "PQC"
        if "HYBRID" in algo:
            return "HYBRID"
        if "RSA" in algo:
            return "RSA"
        if "ECC" in algo:
            return "ECC"
        if "AES" in algo:
            return "AES"
        if "CHACHA" in algo:
            return "CHACHA"
        return "FALLBACK"

    def _pseudo_noise(self, seed: int, amplitude: float = 1.0) -> float:
        """
        Gera ruído determinístico usando função senoidal composta
        Retorna valor entre -amplitude e +amplitude
        """
        x = seed * 0.1
        noise = (
                0.5 * (((seed * 17) % 100) / 100.0 - 0.5) +
                0.3 * (((seed * 31) % 100) / 100.0 - 0.5) +
                0.2 * (((seed * 47) % 100) / 100.0 - 0.5)
        )
        return noise * amplitude

    def _should_match(self, global_step: int, scenario_idx: int, expected_algo: str) -> bool:
        """
        Define se vai dar match, considerando peso do algoritmo
        Algoritmos mais populares têm maior chance de match
        """
        weight = self.algorithm_weights.get(expected_algo, 1.0)

        # Base: ~75-85% de match dependendo do peso
        base_match_rate = 65 + (weight * 10)  # 65% a 83%

        # Hash determinístico
        h = (global_step * 19 + scenario_idx * 13) % 100

        return h < base_match_rate

    def _calculate_success(self, global_step: int, scenario_idx: int,
                           match: bool, expected_algo: str, security_level: str) -> bool:
        """
        Calcula sucesso com base em múltiplos fatores
        - Match/mismatch
        - Tipo de algoritmo
        - Nível de segurança
        - Ruído
        """
        # Taxa base
        if match:
            base_rate = 92  # 92% quando acerta
        else:
            base_rate = 68  # 68% quando erra

        # Ajuste por família de algoritmo
        family = self._algo_family(expected_algo)
        family_bonus = {
            'PQC': 3,  # PQC é mais confiável
            'HYBRID': 2,  # Hybrid também
            'QKD': 0,  # QKD neutro
            'AES': -2,  # AES clássico menos
            'RSA': -2,
            'ECC': -2,
            'CHACHA': -3,
            'FALLBACK': -5  # Fallback pior
        }
        base_rate += family_bonus.get(family, 0)

        # Ajuste por nível de segurança (ultra é mais exigente)
        security_penalty = {
            'ultra': -5,
            'very_high': -3,
            'high': -1,
            'moderate': 0
        }
        base_rate += security_penalty.get(security_level, 0)

        # Adiciona ruído
        noise = self._pseudo_noise(global_step + scenario_idx * 7, amplitude=8)
        final_rate = base_rate + noise

        # Limita entre 10% e 98%
        final_rate = max(10, min(98, final_rate))

        # Decisão determinística
        h = (global_step * 23 + scenario_idx * 11) % 100
        return h < final_rate

    def _calculate_latency(self, expected_algo: str, success: bool,
                           global_step: int, security_level: str) -> float:
        """
        Latência com ruído e variação realista
        """
        latency_map = {
            'QKD': (50, 95),
            'PQC': (28, 58),
            'HYBRID': (55, 105),
            'RSA': (22, 48),
            'ECC': (20, 42),
            'AES': (12, 35),
            'CHACHA': (14, 38),
            'FALLBACK': (18, 45)
        }

        family = self._algo_family(expected_algo)
        base_min, base_max = latency_map.get(family, (30, 60))

        # Variação cíclica
        phase = ((global_step * 7) % 100) / 100.0
        base = base_min + (base_max - base_min) * phase

        # Penalidade por nível de segurança
        security_mult = {
            'ultra': 1.3,
            'very_high': 1.15,
            'high': 1.0,
            'moderate': 0.9
        }
        base *= security_mult.get(security_level, 1.0)

        # Falha = muito mais lento
        if not success:
            base *= 2.8

        # Adiciona ruído significativo
        noise = self._pseudo_noise(global_step * 3, amplitude=base * 0.25)
        latency = base + noise

        # Garante mínimo
        latency = max(5.0, latency)

        return round(latency, 2)

    def _calculate_resource(self, expected_algo: str, success: bool,
                            global_step: int) -> float:
        """
        Uso de recurso com variação
        """
        resource_map = {
            'QKD': 0.85,
            'PQC': 0.68,
            'HYBRID': 0.78,
            'RSA': 0.52,
            'ECC': 0.48,
            'AES': 0.35,
            'CHACHA': 0.38,
            'FALLBACK': 0.32
        }

        family = self._algo_family(expected_algo)
        base = resource_map.get(family, 0.50)

        # Variação
        osc = self._pseudo_noise(global_step * 5, amplitude=0.12)
        value = base + osc

        # Falha consome mais
        if not success:
            value *= 1.15

        value = max(0.05, min(0.99, value))
        return round(value, 2)

    def _calculate_response_time(self, global_step: int, latency: float) -> float:
        """
        Tempo de resposta correlacionado com latência
        """
        # Base proporcional à latência
        base = 0.012 + (latency / 1000.0) * 0.3

        # Ruído
        noise = self._pseudo_noise(global_step * 11, amplitude=0.008)

        response = base + noise
        response = max(0.005, min(0.150, response))

        return round(response, 4)

    def _compute_diversity_penalty(self, algo: str, global_step: int) -> float:
        """
        Penalidade de diversidade mais suave
        """
        total_so_far = sum(self.algorithm_usage_counter.values())
        if total_so_far < 20:
            return 0.0

        # Uso esperado considerando peso
        total_weight = sum(self.algorithm_weights.values())
        algo_weight = self.algorithm_weights.get(algo, 1.0)
        expected = (total_so_far * algo_weight) / total_weight

        used = self.algorithm_usage_counter.get(algo, 0)

        if expected == 0:
            return 0.0

        ratio = used / expected

        # Penalidade suave
        if ratio <= 1.5:
            return 0.0
        elif ratio <= 2.0:
            return 0.03
        elif ratio <= 3.0:
            return 0.08
        else:
            return 0.15

    def _select_proposed_algorithm(self, expected_algo: str, is_match: bool,
                                   global_step: int, scenario_idx: int) -> List[str]:
        """
        Seleciona algoritmo proposto de forma mais realista
        """
        if is_match:
            return [expected_algo]

        # Quando não dá match, escolhe outro baseado em pesos
        # Cria lista ponderada
        candidates = []
        for algo, weight in self.algorithm_weights.items():
            if algo != expected_algo:
                # Adiciona múltiplas vezes baseado no peso
                count = int(weight * 10)
                candidates.extend([algo] * count)

        # Seleciona deterministicamente
        idx = (global_step * 29 + scenario_idx * 37) % len(candidates)
        selected = candidates[idx]

        # 30% de chance de incluir o esperado como fallback
        include_expected = ((global_step + scenario_idx) % 10) < 3

        if include_expected:
            return [selected, expected_algo]
        else:
            return [selected]

    # ------------------------------------------------------------------ #
    # Execução sintética                                                 #
    # ------------------------------------------------------------------ #

    def run_experiment(self, scenarios: List[Dict[str, Any]],
                       episodes: int = 30,
                       iterations_per_episode: int = 50) -> Dict[str, Any]:
        """
        Executa experimento sintético com dados realistas
        """
        total_requests = episodes * iterations_per_episode * len(scenarios)
        print("=" * 80)
        print("SYNTHETIC RL ENGINE EXPERIMENT - REALISTIC VERSION")
        print("=" * 80)
        print(f"Episodes: {episodes}")
        print(f"Iterations per episode: {iterations_per_episode}")
        print(f"Unique scenarios (algorithms): {len(scenarios)}")
        print(f"Total synthetic requests: {total_requests}")
        print("\n🎯 REALISTIC FEATURES:")
        print("  • Variação natural entre algoritmos (alguns mais usados)")
        print("  • Taxa de sucesso ~85-92% (não 100%)")
        print("  • Ruído nos dados de latência e recursos")
        print("  • Falhas ocasionais realistas")
        print("  • Padrões mais naturais e orgânicos")
        print("=" * 80)

        global_step = 0
        experiment_start = time.time()

        for ep in range(1, episodes + 1):
            print(f"\nEPISODE {ep}/{episodes}")
            episode_start = time.time()

            ep_latencies: List[float] = []
            ep_success = 0
            ep_count = 0

            for it in range(1, iterations_per_episode + 1):
                if it % 10 == 0 or it == 1:
                    print(f"  Iteration {it}/{iterations_per_episode}")

                for s_idx, scenario in enumerate(scenarios):
                    global_step += 1
                    ep_count += 1

                    expected_algo = scenario['expected_algorithm']
                    security_level = scenario['context']['security_level']

                    # 1) Match/mismatch com peso
                    is_match = self._should_match(global_step, s_idx, expected_algo)

                    # 2) Algoritmo proposto
                    proposed_algos = self._select_proposed_algorithm(
                        expected_algo, is_match, global_step, s_idx
                    )

                    # Contabiliza uso
                    for p in proposed_algos:
                        self.algorithm_usage_counter[p] = self.algorithm_usage_counter.get(p, 0) + 1

                    # 3) Sucesso com ruído
                    success = self._calculate_success(
                        global_step, s_idx, is_match, expected_algo, security_level
                    )

                    # 4) Métricas com ruído
                    latency = self._calculate_latency(
                        expected_algo, success, global_step, security_level
                    )
                    resource_usage = self._calculate_resource(
                        expected_algo, success, global_step
                    )
                    response_time = self._calculate_response_time(global_step, latency)

                    ep_latencies.append(latency)
                    if success:
                        ep_success += 1

                    # 5) Penalidade de diversidade
                    diversity_penalty = self._compute_diversity_penalty(expected_algo, global_step)

                    result_data = {
                        'scenario_name': scenario['name'],
                        'scenario_category': scenario.get('category', 'general'),
                        'expected_algorithm': expected_algo,
                        'request_id': scenario['context']['request_id'],
                        'security_level': security_level,
                        'risk_score': scenario['context'].get('risk_score'),
                        'conf_score': scenario['context'].get('conf_score'),
                        'has_qkd': 'QKD' in scenario['context'].get('dst_props', {}).get('hardware', []),
                        'proposed_algorithms': proposed_algos,
                        'fallback_algorithms': [],
                        'response_time': response_time,
                        'feedback_success': success,
                        'feedback_latency': latency,
                        'feedback_resource_usage': resource_usage,
                        'diversity_penalty': diversity_penalty,
                        'timestamp': datetime.now().isoformat()
                    }

                    self.results.append(result_data)

            # Métricas do episódio
            episode_elapsed = time.time() - episode_start
            success_rate_ep = (ep_success / ep_count * 100) if ep_count > 0 else 0
            avg_lat_ep = statistics.mean(ep_latencies) if ep_latencies else 0.0

            self.metrics_history.append({
                'episode': ep,
                'elapsed_time': episode_elapsed,
                'episode_requests': ep_count,
                'episode_success_rate': round(success_rate_ep, 2),
                'episode_avg_latency_ms': round(avg_lat_ep, 2)
            })

            print(f"  → Episode {ep} done | "
                  f"success_rate = {success_rate_ep:.2f}% | "
                  f"avg_latency = {avg_lat_ep:.2f} ms")

        experiment_elapsed = time.time() - experiment_start
        print("\n" + "=" * 80)
        print("SYNTHETIC EXPERIMENT COMPLETED")
        print("=" * 80)
        print(f"Total time: {experiment_elapsed:.2f}s ({experiment_elapsed / 60:.2f} minutes)")
        print(f"Total synthetic requests: {len(self.results)}")

        return self.generate_report()

    # ------------------------------------------------------------------ #
    # Relatório                                                          #
    # ------------------------------------------------------------------ #

    def generate_report(self) -> Dict[str, Any]:
        print("\n📊 Generating synthetic report...")

        algorithm_usage: Dict[str, int] = {}
        for result in self.results:
            for algo in result['proposed_algorithms']:
                algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1

        expected_algo_usage: Dict[str, int] = {}
        for result in self.results:
            exp = result['expected_algorithm']
            expected_algo_usage[exp] = expected_algo_usage.get(exp, 0) + 1

        by_security_level: Dict[str, Dict[str, Any]] = {}
        for result in self.results:
            level = result['security_level']
            if level not in by_security_level:
                by_security_level[level] = {
                    'count': 0,
                    'success': 0,
                    'latencies': []
                }
            by_security_level[level]['count'] += 1
            if result['feedback_success']:
                by_security_level[level]['success'] += 1
            by_security_level[level]['latencies'].append(result['feedback_latency'])

        for level in by_security_level:
            data = by_security_level[level]
            data['success_rate'] = (data['success'] / data['count'] * 100) if data['count'] > 0 else 0
            data['avg_latency'] = statistics.mean(data['latencies']) if data['latencies'] else 0
            del data['success']
            del data['latencies']

        qkd_results = [r for r in self.results if r['has_qkd']]
        non_qkd_results = [r for r in self.results if not r['has_qkd']]

        qkd_analysis = {
            'with_qkd': {
                'count': len(qkd_results),
                'success_rate': (sum(1 for r in qkd_results if r['feedback_success']) / len(
                    qkd_results) * 100) if qkd_results else 0,
                'avg_latency': statistics.mean([r['feedback_latency'] for r in qkd_results]) if qkd_results else 0
            },
            'without_qkd': {
                'count': len(non_qkd_results),
                'success_rate': (sum(1 for r in non_qkd_results if r['feedback_success']) / len(
                    non_qkd_results) * 100) if non_qkd_results else 0,
                'avg_latency': statistics.mean(
                    [r['feedback_latency'] for r in non_qkd_results]) if non_qkd_results else 0
            }
        }

        total_requests = len(self.results)
        total_weight = sum(self.algorithm_weights.values())
        expected_per_algo = {
            algo: (total_requests * weight) / total_weight
            for algo, weight in self.algorithm_weights.items()
        }

        diversity_metrics = {
            'total_unique_algorithms': len(algorithm_usage),
            'expected_unique_algorithms': 20,
            'expected_per_algorithm': total_requests / 20,
            'most_used': max(algorithm_usage.items(), key=lambda x: x[1]) if algorithm_usage else ('None', 0),
            'least_used': min(algorithm_usage.items(), key=lambda x: x[1]) if algorithm_usage else ('None', 0),
            'usage_variance': statistics.variance(algorithm_usage.values()) if len(algorithm_usage) > 1 else 0
        }

        successful_requests = sum(1 for r in self.results if r['feedback_success'])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        latencies = [r['feedback_latency'] for r in self.results]
        response_times = [r['response_time'] for r in self.results]

        report = {
            'experiment_info': {
                'total_requests': total_requests,
                'total_episodes': len(self.metrics_history),
                'timestamp': datetime.now().isoformat(),
                'version': 'synthetic-realistic-1.0'
            },
            'performance_metrics': {
                'success_rate': round(success_rate, 2),
                'avg_latency_ms': round(statistics.mean(latencies), 2) if latencies else 0,
                'std_latency_ms': round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
                'min_latency_ms': round(min(latencies), 2) if latencies else 0,
                'max_latency_ms': round(max(latencies), 2) if latencies else 0,
                'avg_response_time_s': round(statistics.mean(response_times), 4) if response_times else 0
            },
            'algorithm_usage': algorithm_usage,
            'expected_algorithm_distribution': expected_algo_usage,
            'diversity_metrics': diversity_metrics,
            'by_security_level': by_security_level,
            'qkd_analysis': qkd_analysis,
            'metrics_history': self.metrics_history,
            'raw_results': self.results
        }

        return report

    def save_results(self, report: Dict[str, Any],
                     prefix: str = "synthetic_rl_experiment"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ JSON Report: {json_file}")

        csv_file = f"{prefix}_{timestamp}_details.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if report['raw_results']:
                writer = csv.DictWriter(f, fieldnames=report['raw_results'][0].keys())
                writer.writeheader()
                writer.writerows(report['raw_results'])
        print(f"✅ Details CSV: {csv_file}")

        txt_file = f"{prefix}_{timestamp}_summary.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SYNTHETIC RL ENGINE EXPERIMENT - REALISTIC\n")
            f.write("=" * 80 + "\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            for key, value in report['performance_metrics'].items():
                f.write(f"{key}: {value}\n")

            f.write("\nDIVERSITY METRICS\n")
            f.write("-" * 80 + "\n")
            dm = report['diversity_metrics']
            f.write(
                f"Total unique algorithms used: {dm['total_unique_algorithms']}/{dm['expected_unique_algorithms']}\n")
            f.write(f"Expected per algorithm: {dm['expected_per_algorithm']:.1f}\n")
            f.write(f"Most used: {dm['most_used'][0]} ({dm['most_used'][1]} times)\n")
            f.write(f"Least used: {dm['least_used'][0]} ({dm['least_used'][1]} times)\n")
            f.write(f"Usage variance: {dm['usage_variance']:.2f}\n")

            f.write("\nEXPECTED ALGORITHM DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            for algo, count in sorted(report['expected_algorithm_distribution'].items()):
                percentage = (count / report['experiment_info']['total_requests']) * 100 \
                    if report['experiment_info']['total_requests'] > 0 else 0
                f.write(f"{algo}: {count} requests ({percentage:.1f}%)\n")

            f.write("\nACTUAL ALGORITHM USAGE (Proposed by synthetic engine)\n")
            f.write("-" * 80 + "\n")
            sorted_algos = sorted(report['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for algo, count in sorted_algos:
                percentage = (count / report['experiment_info']['total_requests']) * 100 \
                    if report['experiment_info']['total_requests'] > 0 else 0
                deviation = count - dm['expected_per_algorithm']
                f.write(f"{algo}: {count} times ({percentage:.1f}%) "
                        f"[deviation: {deviation:+.1f}]\n")

            f.write("\nBY SECURITY LEVEL\n")
            f.write("-" * 80 + "\n")
            for level, data in sorted(report['by_security_level'].items()):
                f.write(f"\n{level.upper()}:\n")
                f.write(f"  Requests: {data['count']}\n")
                f.write(f"  Success Rate: {data['success_rate']:.2f}%\n")
                f.write(f"  Avg Latency: {data['avg_latency']:.2f}ms\n")

            f.write("\nQKD HARDWARE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"With QKD Hardware:\n")
            f.write(f"  Requests: {report['qkd_analysis']['with_qkd']['count']}\n")
            f.write(f"  Success Rate: {report['qkd_analysis']['with_qkd']['success_rate']:.2f}%\n")
            f.write(f"  Avg Latency: {report['qkd_analysis']['with_qkd']['avg_latency']:.2f}ms\n\n")
            f.write(f"Without QKD Hardware:\n")
            f.write(f"  Requests: {report['qkd_analysis']['without_qkd']['count']}\n")
            f.write(f"  Success Rate: {report['qkd_analysis']['without_qkd']['success_rate']:.2f}%\n")
            f.write(f"  Avg Latency: {report['qkd_analysis']['without_qkd']['avg_latency']:.2f}ms\n")

        print(f"✅ Summary TXT: {txt_file}")

        return {
            'json': json_file,
            'csv': csv_file,
            'summary': txt_file
        }


# ---------------------------------------------------------------------- #
# Cenários sintéticos                                                    #
# ---------------------------------------------------------------------- #

SYNTHETIC_SCENARIOS = [
    {
        'name': 'QKD BB84 - Ultra Security',
        'category': 'quantum_bb84',
        'expected_algorithm': 'QKD_BB84',
        'context': {
            'request_id': 'qkd-bb84-001',
            'source': 'synthetic-quantum-bb84-node',
            'destination': 'http://synthetic-secure-endpoint',
            'security_level': 'ultra',
            'risk_score': 0.95,
            'conf_score': 0.98,
            'data_sensitivity': 0.98,
            'network_latency': 0.15,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['FIPS-140-3', 'QUANTUM-SAFE']
            }
        }
    },
    {
        'name': 'QKD E91 - Entanglement Based',
        'category': 'quantum_e91',
        'expected_algorithm': 'QKD_E91',
        'context': {
            'request_id': 'qkd-e91-002',
            'source': 'synthetic-quantum-e91-node',
            'destination': 'http://synthetic-quantum-hub',
            'security_level': 'very_high',
            'risk_score': 0.90,
            'conf_score': 0.93,
            'data_sensitivity': 0.92,
            'network_latency': 0.18,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM', 'ENTANGLEMENT'],
                'compliance': ['ISO27001', 'QUANTUM-SAFE']
            }
        }
    },
    {
        'name': 'QKD CV-QKD - Continuous Variable',
        'category': 'quantum_cv',
        'expected_algorithm': 'QKD_CV-QKD',
        'context': {
            'request_id': 'qkd-cv-003',
            'source': 'synthetic-quantum-cv-node',
            'destination': 'http://synthetic-cv-quantum-server',
            'security_level': 'high',
            'risk_score': 0.85,
            'conf_score': 0.88,
            'data_sensitivity': 0.87,
            'network_latency': 0.12,
            'dst_props': {
                'hardware': ['QKD', 'CV-QUANTUM'],
                'compliance': ['GDPR', 'QUANTUM-SAFE']
            }
        }
    },
    {
        'name': 'QKD MDI-QKD - Measurement Device Independent',
        'category': 'quantum_mdi',
        'expected_algorithm': 'QKD_MDI-QKD',
        'context': {
            'request_id': 'qkd-mdi-004',
            'source': 'synthetic-quantum-mdi-node',
            'destination': 'http://synthetic-mdi-quantum-relay',
            'security_level': 'high',
            'risk_score': 0.83,
            'conf_score': 0.86,
            'data_sensitivity': 0.85,
            'network_latency': 0.20,
            'dst_props': {
                'hardware': ['QKD', 'MDI-QUANTUM'],
                'compliance': ['SOC2', 'QUANTUM-SAFE']
            }
        }
    },
    {
        'name': 'QKD DECOY - Decoy State Protocol',
        'category': 'quantum_decoy',
        'expected_algorithm': 'QKD_DECOY',
        'context': {
            'request_id': 'qkd-decoy-005',
            'source': 'synthetic-quantum-decoy-node',
            'destination': 'http://synthetic-decoy-quantum-node',
            'security_level': 'very_high',
            'risk_score': 0.91,
            'conf_score': 0.94,
            'data_sensitivity': 0.93,
            'network_latency': 0.16,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM', 'DECOY-STATE'],
                'compliance': ['NIST-PQC', 'QUANTUM-SAFE']
            }
        }
    },
    {
        'name': 'PQC KYBER - Post-Quantum KEM',
        'category': 'pqc_kyber',
        'expected_algorithm': 'PQC_KYBER',
        'context': {
            'request_id': 'pqc-kyber-006',
            'source': 'synthetic-pqc-kyber-node',
            'destination': 'http://synthetic-pqc-server-01',
            'security_level': 'ultra',
            'risk_score': 0.93,
            'conf_score': 0.96,
            'data_sensitivity': 0.95,
            'network_latency': 0.08,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR'],
                'compliance': ['NIST-PQC', 'FIPS-140-3']
            }
        }
    },
    {
        'name': 'PQC DILITHIUM - Digital Signature',
        'category': 'pqc_dilithium',
        'expected_algorithm': 'PQC_DILITHIUM',
        'context': {
            'request_id': 'pqc-dilithium-007',
            'source': 'synthetic-pqc-dilithium-node',
            'destination': 'http://synthetic-signature-server',
            'security_level': 'very_high',
            'risk_score': 0.88,
            'conf_score': 0.91,
            'data_sensitivity': 0.90,
            'network_latency': 0.10,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR'],
                'compliance': ['FIPS-140-3', 'NIST-PQC']
            }
        }
    },
    {
        'name': 'PQC NTRU - Lattice-Based Encryption',
        'category': 'pqc_ntru',
        'expected_algorithm': 'PQC_NTRU',
        'context': {
            'request_id': 'pqc-ntru-008',
            'source': 'synthetic-pqc-ntru-node',
            'destination': 'http://synthetic-lattice-crypto-server',
            'security_level': 'very_high',
            'risk_score': 0.86,
            'conf_score': 0.89,
            'data_sensitivity': 0.88,
            'network_latency': 0.07,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR'],
                'compliance': ['GDPR', 'NIST-PQC']
            }
        }
    },
    {
        'name': 'PQC SABER - Module Learning',
        'category': 'pqc_saber',
        'expected_algorithm': 'PQC_SABER',
        'context': {
            'request_id': 'pqc-saber-009',
            'source': 'synthetic-pqc-saber-node',
            'destination': 'http://synthetic-saber-endpoint',
            'security_level': 'high',
            'risk_score': 0.80,
            'conf_score': 0.84,
            'data_sensitivity': 0.82,
            'network_latency': 0.09,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR'],
                'compliance': ['ISO27001', 'NIST-PQC']
            }
        }
    },
    {
        'name': 'PQC FALCON - Fast Fourier Lattice',
        'category': 'pqc_falcon',
        'expected_algorithm': 'PQC_FALCON',
        'context': {
            'request_id': 'pqc-falcon-010',
            'source': 'synthetic-pqc-falcon-node',
            'destination': 'http://synthetic-falcon-crypto-hub',
            'security_level': 'ultra',
            'risk_score': 0.94,
            'conf_score': 0.97,
            'data_sensitivity': 0.96,
            'network_latency': 0.11,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR', 'FFT-UNIT'],
                'compliance': ['NIST-PQC', 'FIPS-140-3']
            }
        }
    },
    {
        'name': 'PQC SPHINCS+ - Stateless Hash-Based',
        'category': 'pqc_sphincs',
        'expected_algorithm': 'PQC_SPHINCS',
        'context': {
            'request_id': 'pqc-sphincs-011',
            'source': 'synthetic-pqc-sphincs-node',
            'destination': 'http://synthetic-hash-signature-server',
            'security_level': 'very_high',
            'risk_score': 0.89,
            'conf_score': 0.92,
            'data_sensitivity': 0.91,
            'network_latency': 0.13,
            'dst_props': {
                'hardware': ['HASH-ACCELERATOR'],
                'compliance': ['NIST-PQC', 'FIPS-140-3']
            }
        }
    },
    {
        'name': 'HYBRID QKD+PQC - Maximum Security',
        'category': 'hybrid_qkd_pqc',
        'expected_algorithm': 'HYBRID_QKD_PQC',
        'context': {
            'request_id': 'hybrid-qkd-pqc-012',
            'source': 'synthetic-hybrid-quantum-pqc-node',
            'destination': 'http://synthetic-ultra-secure-vault',
            'security_level': 'ultra',
            'risk_score': 0.97,
            'conf_score': 0.99,
            'data_sensitivity': 0.98,
            'network_latency': 0.22,
            'dst_props': {
                'hardware': ['QKD', 'PQC-ACCELERATOR', 'QUANTUM'],
                'compliance': ['FIPS-140-3', 'NIST-PQC', 'QUANTUM-SAFE']
            }
        }
    },
    {
        'name': 'HYBRID RSA+PQC - Transition Security',
        'category': 'hybrid_rsa',
        'expected_algorithm': 'HYBRID_RSA_PQC',
        'context': {
            'request_id': 'hybrid-rsa-013',
            'source': 'synthetic-hybrid-rsa-node',
            'destination': 'http://synthetic-transition-server',
            'security_level': 'very_high',
            'risk_score': 0.82,
            'conf_score': 0.85,
            'data_sensitivity': 0.84,
            'network_latency': 0.14,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR', 'RSA-COPROCESSOR'],
                'compliance': ['SOC2', 'NIST-PQC']
            }
        }
    },
    {
        'name': 'HYBRID ECC+PQC - Elliptic Curve Hybrid',
        'category': 'hybrid_ecc',
        'expected_algorithm': 'HYBRID_ECC_PQC',
        'context': {
            'request_id': 'hybrid-ecc-014',
            'source': 'synthetic-hybrid-ecc-node',
            'destination': 'http://synthetic-ecc-hybrid-endpoint',
            'security_level': 'high',
            'risk_score': 0.75,
            'conf_score': 0.79,
            'data_sensitivity': 0.77,
            'network_latency': 0.11,
            'dst_props': {
                'hardware': ['PQC-ACCELERATOR', 'ECC-UNIT'],
                'compliance': ['GDPR', 'NIST-PQC']
            }
        }
    },
    {
        'name': 'RSA 4096 - Classical Strong',
        'category': 'classical_rsa',
        'expected_algorithm': 'RSA_4096',
        'context': {
            'request_id': 'rsa-4096-015',
            'source': 'synthetic-classical-rsa-node',
            'destination': 'http://synthetic-legacy-secure-server',
            'security_level': 'high',
            'risk_score': 0.70,
            'conf_score': 0.74,
            'data_sensitivity': 0.72,
            'network_latency': 0.08,
            'dst_props': {
                'hardware': ['RSA-COPROCESSOR'],
                'compliance': ['FIPS-140-2', 'PCI-DSS']
            }
        }
    },
    {
        'name': 'ECC 521 - Elliptic Curve',
        'category': 'classical_ecc',
        'expected_algorithm': 'ECC_521',
        'context': {
            'request_id': 'ecc-521-016',
            'source': 'synthetic-classical-ecc-node',
            'destination': 'http://synthetic-ecc-server',
            'security_level': 'high',
            'risk_score': 0.72,
            'conf_score': 0.76,
            'data_sensitivity': 0.74,
            'network_latency': 0.06,
            'dst_props': {
                'hardware': ['ECC-UNIT'],
                'compliance': ['PCI-DSS', 'ISO27001']
            }
        }
    },
    {
        'name': 'AES 256 GCM - Symmetric Encryption',
        'category': 'classical_aes256',
        'expected_algorithm': 'AES_256_GCM',
        'context': {
            'request_id': 'aes-256-017',
            'source': 'synthetic-classical-aes256-node',
            'destination': 'http://synthetic-standard-server',
            'security_level': 'moderate',
            'risk_score': 0.55,
            'conf_score': 0.60,
            'data_sensitivity': 0.58,
            'network_latency': 0.05,
            'dst_props': {
                'hardware': ['AES-NI'],
                'compliance': ['FIPS-140-2']
            }
        }
    },
    {
        'name': 'AES 192 - Medium Symmetric',
        'category': 'classical_aes192',
        'expected_algorithm': 'AES_192',
        'context': {
            'request_id': 'aes-192-018',
            'source': 'synthetic-classical-aes192-node',
            'destination': 'http://synthetic-medium-security-server',
            'security_level': 'moderate',
            'risk_score': 0.50,
            'conf_score': 0.55,
            'data_sensitivity': 0.53,
            'network_latency': 0.04,
            'dst_props': {
                'hardware': ['AES-NI'],
                'compliance': []
            }
        }
    },
    {
        'name': 'ChaCha20-Poly1305 - Stream Cipher',
        'category': 'classical_chacha',
        'expected_algorithm': 'CHACHA20_POLY1305',
        'context': {
            'request_id': 'chacha20-019',
            'source': 'synthetic-classical-chacha-node',
            'destination': 'http://synthetic-mobile-optimized-server',
            'security_level': 'moderate',
            'risk_score': 0.52,
            'conf_score': 0.57,
            'data_sensitivity': 0.55,
            'network_latency': 0.05,
            'dst_props': {
                'hardware': ['MOBILE-CRYPTO'],
                'compliance': []
            }
        }
    },
    {
        'name': 'FALLBACK AES - Emergency Mode',
        'category': 'fallback',
        'expected_algorithm': 'FALLBACK_AES',
        'context': {
            'request_id': 'fallback-aes-020',
            'source': 'synthetic-fallback-node',
            'destination': 'http://synthetic-emergency-server',
            'security_level': 'moderate',
            'risk_score': 0.60,
            'conf_score': 0.65,
            'data_sensitivity': 0.62,
            'available_resources': 0.20,
            'system_load': 0.95,
            'network_latency': 0.25,
            'dst_props': {
                'hardware': [],
                'compliance': []
            }
        }
    }
]


def main():
    print("\n" + "=" * 80)
    print("SYNTHETIC RL ENGINE EXPERIMENT - REALISTIC VERSION")
    print("=" * 80)

    episodes = 30
    iterations_per_episode = 50

    experiment = SyntheticRLExperiment()
    report = experiment.run_experiment(
        scenarios=SYNTHETIC_SCENARIOS,
        episodes=episodes,
        iterations_per_episode=iterations_per_episode
    )

    if report:
        files = experiment.save_results(report, prefix="synthetic_rl_realistic")

        print("\n" + "=" * 80)
        print("GENERATED FILES:")
        print("=" * 80)
        for file_type, filename in files.items():
            print(f"  {file_type}: {filename}")

        print("\n" + "=" * 80)
        print("QUICK SUMMARY:")
        print("=" * 80)
        print(f"Success rate: {report['performance_metrics']['success_rate']:.2f}%")
        print(f"Average latency: {report['performance_metrics']['avg_latency_ms']:.2f}ms")
        print(f"Latency std dev: {report['performance_metrics']['std_latency_ms']:.2f}ms")
        print(f"Total requests: {report['experiment_info']['total_requests']}")

        dm = report['diversity_metrics']
        print(f"\n📊 DIVERSITY ANALYSIS:")
        print(f"  Unique algorithms used: {dm['total_unique_algorithms']}/{dm['expected_unique_algorithms']}")
        print(f"  Most used: {dm['most_used'][0]} ({dm['most_used'][1]} times)")
        print(f"  Least used: {dm['least_used'][0]} ({dm['least_used'][1]} times)")
        print(f"  Usage variance: {dm['usage_variance']:.2f}")

        print("\n✅ Realistic synthetic experiment completed!")


if __name__ == "__main__":
    main()