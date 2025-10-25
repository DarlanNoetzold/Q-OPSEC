import requests
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import statistics
import random

class UltraBalancedExperiment:
    def __init__(self, base_url: str = "http://localhost:9009"):
        self.base_url = base_url
        self.results = []
        self.metrics_history = []

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def enable_training(self):
        response = requests.post(f"{self.base_url}/training/enable")
        return response.json()

    def disable_training(self):
        response = requests.post(f"{self.base_url}/training/disable")
        return response.json()

    def get_metrics(self) -> Dict:
        response = requests.get(f"{self.base_url}/metrics")
        return response.json()

    def end_episode(self) -> Dict:
        response = requests.post(f"{self.base_url}/episode/end")
        return response.json()

    def send_request(self, context: Dict) -> Dict:
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/act",
            json=context,
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = time.time() - start_time

        result = response.json()
        result['response_time'] = elapsed_time
        result['timestamp'] = datetime.now().isoformat()

        return result

    def send_feedback(self, request_id: str, success: bool,
                      latency: float, resource_usage: float):
        feedback = {
            "request_id": request_id,
            "success": success,
            "latency": latency,
            "resource_usage": resource_usage
        }
        response = requests.post(
            f"{self.base_url}/feedback",
            json=feedback,
            headers={"Content-Type": "application/json"}
        )
        return response.json()

    def generate_feedback_for_algorithm(self, expected_algo: str,
                                        proposed_algos: List[str],
                                        scenario: Dict) -> Dict:
        """Generate feedback with EXTREME rewards for correct algorithm selection"""

        # Check if expected algorithm was proposed
        algo_match = any(expected_algo in algo for algo in proposed_algos)

        # EXTREME reward/penalty system to force learning
        if algo_match:
            # EXTREME REWARD: Almost guaranteed success
            base_success = 0.99
            latency_multiplier = 0.7
            resource_multiplier = 0.8
        else:
            # EXTREME PENALTY: Almost guaranteed failure
            base_success = 0.20
            latency_multiplier = 3.5
            resource_multiplier = 2.0

        # Add minimal randomness to maintain determinism
        success = random.random() < base_success

        # Algorithm-specific latencies (highly optimized ranges)
        latency_map = {
            'BB84': (40, 60),
            'E91': (42, 62),
            'CV-QKD': (38, 58),
            'MDI-QKD': (45, 65),
            'DECOY': (43, 63),
            'KYBER': (20, 35),
            'DILITHIUM': (22, 37),
            'NTRU': (18, 33),
            'SABER': (19, 34),
            'FALCON': (21, 36),
            'SPHINCS': (25, 40),
            'RSA': (15, 30),
            'ECC': (13, 28),
            'AES': (10, 25),
            'CHACHA': (11, 26),
            'HYBRID': (25, 45)
        }

        # Find latency range
        latency_range = (20, 40)  # default
        for key, value in latency_map.items():
            if key in expected_algo:
                latency_range = value
                break

        if success:
            latency = random.uniform(latency_range[0], latency_range[1]) * latency_multiplier
        else:
            latency = random.uniform(latency_range[1] * 2.0, latency_range[1] * 3.0)

        # Resource usage with extreme differentiation
        resource_map = {
            'QKD': 0.85,
            'PQC': 0.65,
            'HYBRID': 0.75,
            'RSA': 0.50,
            'ECC': 0.45,
            'AES': 0.35,
            'CHACHA': 0.37,
            'FALLBACK': 0.30
        }

        resource_usage = 0.50
        for key, value in resource_map.items():
            if key in expected_algo:
                resource_usage = value
                break

        resource_usage = resource_usage * resource_multiplier

        return {
            'success': success,
            'latency': round(latency, 2),
            'resource_usage': round(min(1.0, resource_usage + random.uniform(-0.02, 0.02)), 2)
        }

    def run_scenario(self, scenario: Dict):
        """Execute a test scenario"""
        print(f"  → {scenario['name']}")

        result = self.send_request(scenario['context'])

        proposed_algos = result.get('payload', {}).get('proposed', [])
        expected_algo = scenario.get('expected_algorithm', '')

        feedback = self.generate_feedback_for_algorithm(
            expected_algo,
            proposed_algos,
            scenario
        )

        time.sleep(0.1)  # Reduced to 100ms

        self.send_feedback(
            scenario['context']['request_id'],
            feedback['success'],
            feedback['latency'],
            feedback['resource_usage']
        )

        result_data = {
            'scenario_name': scenario['name'],
            'scenario_category': scenario.get('category', 'general'),
            'expected_algorithm': expected_algo,
            'request_id': scenario['context']['request_id'],
            'security_level': scenario['context']['security_level'],
            'risk_score': scenario['context'].get('risk_score'),
            'conf_score': scenario['context'].get('conf_score'),
            'has_qkd': 'QKD' in scenario['context'].get('dst_props', {}).get('hardware', []),
            'proposed_algorithms': proposed_algos,
            'fallback_algorithms': result.get('payload', {}).get('fallback', []),
            'response_time': result.get('response_time'),
            'feedback_success': feedback['success'],
            'feedback_latency': feedback['latency'],
            'feedback_resource_usage': feedback['resource_usage'],
            'timestamp': result.get('timestamp')
        }

        self.results.append(result_data)

        status = "✓ SUCCESS" if feedback['success'] else "✗ FAILED"
        algo_match = "✓" if any(expected_algo in a for a in proposed_algos) else "✗"
        print(f"    {status} | Expected: {expected_algo} {algo_match} | Latency: {feedback['latency']:.2f}ms")

        return result_data

    def run_experiment(self, scenarios: List[Dict], episodes: int = 10,
                       iterations_per_episode: int = 5):
        """Execute complete experiment - ULTRA BALANCED VERSION"""
        print("=" * 80)
        print("RL ENGINE - ULTRA BALANCED EXPERIMENT v7.0")
        print("=" * 80)
        print(f"Episodes: {episodes}")
        print(f"Iterations per episode: {iterations_per_episode}")
        print(f"Unique scenarios: {len(scenarios)}")
        print(f"Total requests: {episodes * iterations_per_episode * len(scenarios)}")
        print(f"Expected per algorithm: {episodes * iterations_per_episode} requests")
        print(f"Estimated time: ~{(episodes * iterations_per_episode * len(scenarios) * 0.15 / 60):.1f} minutes")

        if not self.health_check():
            print("\n❌ ERROR: RL Engine is not responding!")
            return

        print("\n✅ RL Engine is online!")

        self.enable_training()
        print("✅ Training mode enabled")

        experiment_start = time.time()

        for episode in range(1, episodes + 1):
            print(f"\n{'=' * 80}")
            print(f"EPISODE {episode}/{episodes}")
            print(f"{'=' * 80}")

            episode_start = time.time()

            for iteration in range(iterations_per_episode):
                print(f"\n[Iteration {iteration + 1}/{iterations_per_episode}]")

                # Shuffle scenarios to avoid patterns
                shuffled_scenarios = random.sample(scenarios, len(scenarios))

                for idx, scenario in enumerate(shuffled_scenarios, 1):
                    print(f"  [{idx}/{len(scenarios)}]", end=" ")
                    self.run_scenario(scenario)

                if iteration < iterations_per_episode - 1:
                    time.sleep(0.3)

            episode_result = self.end_episode()
            episode_elapsed = time.time() - episode_start

            metrics = self.get_metrics()
            metrics['episode'] = episode
            metrics['elapsed_time'] = episode_elapsed
            self.metrics_history.append(metrics)

            print(f"\n  ✅ Episode {episode} completed in {episode_elapsed:.2f}s")

            if episode < episodes:
                time.sleep(0.8)

        experiment_elapsed = time.time() - experiment_start

        print(f"\n{'=' * 80}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'=' * 80}")
        print(f"Total time: {experiment_elapsed:.2f}s ({experiment_elapsed / 60:.2f} minutes)")
        print(f"Total requests: {len(self.results)}")

        self.disable_training()
        print("✅ Training mode disabled")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate complete report in English"""
        print("\n📊 Generating report...")

        algorithm_usage = {}
        for result in self.results:
            for algo in result['proposed_algorithms']:
                algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1

        expected_algo_usage = {}
        for result in self.results:
            exp = result['expected_algorithm']
            expected_algo_usage[exp] = expected_algo_usage.get(exp, 0) + 1

        # By security level
        by_security_level = {}
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

        # Calculate metrics by security level
        for level in by_security_level:
            data = by_security_level[level]
            data['success_rate'] = (data['success'] / data['count'] * 100) if data['count'] > 0 else 0
            data['avg_latency'] = statistics.mean(data['latencies']) if data['latencies'] else 0
            del data['success']
            del data['latencies']

        # QKD analysis
        qkd_results = [r for r in self.results if r['has_qkd']]
        non_qkd_results = [r for r in self.results if not r['has_qkd']]

        qkd_analysis = {
            'with_qkd': {
                'count': len(qkd_results),
                'success_rate': (sum(1 for r in qkd_results if r['feedback_success']) / len(qkd_results) * 100) if qkd_results else 0,
                'avg_latency': statistics.mean([r['feedback_latency'] for r in qkd_results]) if qkd_results else 0
            },
            'without_qkd': {
                'count': len(non_qkd_results),
                'success_rate': (sum(1 for r in non_qkd_results if r['feedback_success']) / len(non_qkd_results) * 100) if non_qkd_results else 0,
                'avg_latency': statistics.mean([r['feedback_latency'] for r in non_qkd_results]) if non_qkd_results else 0
            }
        }

        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['feedback_success'])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        latencies = [r['feedback_latency'] for r in self.results]
        response_times = [r['response_time'] for r in self.results]

        report = {
            'experiment_info': {
                'total_requests': total_requests,
                'total_episodes': len(self.metrics_history),
                'timestamp': datetime.now().isoformat(),
                'version': '7.0'
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
            'by_security_level': by_security_level,
            'qkd_analysis': qkd_analysis,
            'metrics_history': self.metrics_history,
            'raw_results': self.results
        }

        return report

    def save_results(self, report: Dict, prefix: str = "rl_experiment_v7"):
        """Save results in English"""
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
            f.write("RL ENGINE - ULTRA BALANCED EXPERIMENT v7.0\n")
            f.write("=" * 80 + "\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            for key, value in report['performance_metrics'].items():
                f.write(f"{key}: {value}\n")

            f.write("\nEXPECTED ALGORITHM DISTRIBUTION (Each should have 50 requests)\n")
            f.write("-" * 80 + "\n")
            for algo, count in sorted(report['expected_algorithm_distribution'].items()):
                percentage = (count / report['experiment_info']['total_requests']) * 100
                f.write(f"{algo}: {count} requests ({percentage:.1f}%)\n")

            f.write("\nACTUAL ALGORITHM USAGE (Proposed by RL Engine)\n")
            f.write("-" * 80 + "\n")
            sorted_algos = sorted(report['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for algo, count in sorted_algos:
                percentage = (count / report['experiment_info']['total_requests']) * 100
                f.write(f"{algo}: {count} times ({percentage:.1f}%)\n")

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


# ULTRA BALANCED SCENARIOS - 20 algorithms with varied contexts
ULTRA_BALANCED_SCENARIOS = [
    # === 1. QKD BB84 (5%) ===
    {
        'name': 'QKD BB84 - Ultra Security',
        'category': 'quantum_bb84',
        'expected_algorithm': 'QKD_BB84',
        'context': {
            'request_id': 'qkd-bb84-001',
            'source': 'quantum-bb84-node',
            'destination': 'http://secure-endpoint:9000',
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

    # === 2. QKD E91 (5%) ===
    {
        'name': 'QKD E91 - Entanglement Based',
        'category': 'quantum_e91',
        'expected_algorithm': 'QKD_E91',
        'context': {
            'request_id': 'qkd-e91-002',
            'source': 'quantum-e91-node',
            'destination': 'http://quantum-hub:9000',
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

    # === 3. QKD CV-QKD (5%) ===
    {
        'name': 'QKD CV-QKD - Continuous Variable',
        'category': 'quantum_cv',
        'expected_algorithm': 'QKD_CV-QKD',
        'context': {
            'request_id': 'qkd-cv-003',
            'source': 'quantum-cv-node',
            'destination': 'http://cv-quantum-server:9000',
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

    # === 4. QKD MDI-QKD (5%) ===
    {
        'name': 'QKD MDI-QKD - Measurement Device Independent',
        'category': 'quantum_mdi',
        'expected_algorithm': 'QKD_MDI-QKD',
        'context': {
            'request_id': 'qkd-mdi-004',
            'source': 'quantum-mdi-node',
            'destination': 'http://mdi-quantum-relay:9000',
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

    # === 5. QKD DECOY (5%) ===
    {
        'name': 'QKD DECOY - Decoy State Protocol',
        'category': 'quantum_decoy',
        'expected_algorithm': 'QKD_DECOY',
        'context': {
            'request_id': 'qkd-decoy-005',
            'source': 'quantum-decoy-node',
            'destination': 'http://decoy-quantum-node:9000',
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

    # === 6. PQC KYBER (5%) ===
    {
        'name': 'PQC KYBER - Post-Quantum KEM',
        'category': 'pqc_kyber',
        'expected_algorithm': 'PQC_KYBER',
        'context': {
            'request_id': 'pqc-kyber-006',
            'source': 'pqc-kyber-node',
            'destination': 'http://pqc-server-01:9000',
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

    # === 7. PQC DILITHIUM (5%) ===
    {
        'name': 'PQC DILITHIUM - Digital Signature',
        'category': 'pqc_dilithium',
        'expected_algorithm': 'PQC_DILITHIUM',
        'context': {
            'request_id': 'pqc-dilithium-007',
            'source': 'pqc-dilithium-node',
            'destination': 'http://signature-server:9000',
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

    # === 8. PQC NTRU (5%) ===
    {
        'name': 'PQC NTRU - Lattice-Based Encryption',
        'category': 'pqc_ntru',
        'expected_algorithm': 'PQC_NTRU',
        'context': {
            'request_id': 'pqc-ntru-008',
            'source': 'pqc-ntru-node',
            'destination': 'http://lattice-crypto-server:9000',
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

    # === 9. PQC SABER (5%) ===
    {
        'name': 'PQC SABER - Module Learning',
        'category': 'pqc_saber',
        'expected_algorithm': 'PQC_SABER',
        'context': {
            'request_id': 'pqc-saber-009',
            'source': 'pqc-saber-node',
            'destination': 'http://saber-endpoint:9000',
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

    # === 10. PQC FALCON (5%) ===
    {
        'name': 'PQC FALCON - Fast Fourier Lattice',
        'category': 'pqc_falcon',
        'expected_algorithm': 'PQC_FALCON',
        'context': {
            'request_id': 'pqc-falcon-010',
            'source': 'pqc-falcon-node',
            'destination': 'http://falcon-crypto-hub:9000',
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

    # === 11. PQC SPHINCS+ (5%) ===
    {
        'name': 'PQC SPHINCS+ - Stateless Hash-Based',
        'category': 'pqc_sphincs',
        'expected_algorithm': 'PQC_SPHINCS',
        'context': {
            'request_id': 'pqc-sphincs-011',
            'source': 'pqc-sphincs-node',
            'destination': 'http://hash-signature-server:9000',
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

    # === 12. HYBRID QKD+PQC (5%) ===
    {
        'name': 'HYBRID QKD+PQC - Maximum Security',
        'category': 'hybrid_qkd_pqc',
        'expected_algorithm': 'HYBRID_QKD_PQC',
        'context': {
            'request_id': 'hybrid-qkd-pqc-012',
            'source': 'hybrid-quantum-pqc-node',
            'destination': 'http://ultra-secure-vault:9000',
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

    # === 13. HYBRID RSA+PQC (5%) ===
    {
        'name': 'HYBRID RSA+PQC - Transition Security',
        'category': 'hybrid_rsa',
        'expected_algorithm': 'HYBRID_RSA_PQC',
        'context': {
            'request_id': 'hybrid-rsa-013',
            'source': 'hybrid-rsa-node',
            'destination': 'http://transition-server:9000',
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

    # === 14. HYBRID ECC+PQC (5%) ===
    {
        'name': 'HYBRID ECC+PQC - Elliptic Curve Hybrid',
        'category': 'hybrid_ecc',
        'expected_algorithm': 'HYBRID_ECC_PQC',
        'context': {
            'request_id': 'hybrid-ecc-014',
            'source': 'hybrid-ecc-node',
            'destination': 'http://ecc-hybrid-endpoint:9000',
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

    # === 15. RSA 4096 (5%) ===
    {
        'name': 'RSA 4096 - Classical Strong',
        'category': 'classical_rsa',
        'expected_algorithm': 'RSA_4096',
        'context': {
            'request_id': 'rsa-4096-015',
            'source': 'classical-rsa-node',
            'destination': 'http://legacy-secure-server:9000',
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

    # === 16. ECC 521 (5%) ===
    {
        'name': 'ECC 521 - Elliptic Curve',
        'category': 'classical_ecc',
        'expected_algorithm': 'ECC_521',
        'context': {
            'request_id': 'ecc-521-016',
            'source': 'classical-ecc-node',
            'destination': 'http://ecc-server:9000',
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

    # === 17. AES 256 GCM (5%) ===
    {
        'name': 'AES 256 GCM - Symmetric Encryption',
        'category': 'classical_aes256',
        'expected_algorithm': 'AES_256_GCM',
        'context': {
            'request_id': 'aes-256-017',
            'source': 'classical-aes256-node',
            'destination': 'http://standard-server:9000',
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

    # === 18. AES 192 (5%) ===
    {
        'name': 'AES 192 - Medium Symmetric',
        'category': 'classical_aes192',
        'expected_algorithm': 'AES_192',
        'context': {
            'request_id': 'aes-192-018',
            'source': 'classical-aes192-node',
            'destination': 'http://medium-security-server:9000',
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

    # === 19. ChaCha20-Poly1305 (5%) ===
    {
        'name': 'ChaCha20-Poly1305 - Stream Cipher',
        'category': 'classical_chacha',
        'expected_algorithm': 'CHACHA20_POLY1305',
        'context': {
            'request_id': 'chacha20-019',
            'source': 'classical-chacha-node',
            'destination': 'http://mobile-optimized-server:9000',
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

    # === 20. FALLBACK AES (5%) ===
    {
        'name': 'FALLBACK AES - Emergency Mode',
        'category': 'fallback',
        'expected_algorithm': 'FALLBACK_AES',
        'context': {
            'request_id': 'fallback-aes-020',
            'source': 'fallback-node',
            'destination': 'http://emergency-server:9000',
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
    print("RL ENGINE - ULTRA BALANCED EXPERIMENT v7.0")
    print("=" * 80)
    print("\n📊 DISTRIBUTION PLAN:")
    print(f"  • 20 unique algorithms")
    print(f"  • Each algorithm: 5% of total requests")
    print(f"  • 10 episodes × 5 iterations × 20 scenarios = 1000 total requests")
    print(f"  • Expected per algorithm: 50 requests (exactly 5%)")
    print(f"  • EXTREME reward system: 99% success for correct, 20% for wrong")
    print(f"  • Varied contexts to encourage exploration")

    experiment = UltraBalancedExperiment(base_url="http://localhost:9009")

    report = experiment.run_experiment(
        scenarios=ULTRA_BALANCED_SCENARIOS,
        episodes=10,
        iterations_per_episode=5
    )

    if report:
        files = experiment.save_results(report, prefix="rl_experiment_v7_ultra")

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
        print(f"Total requests: {report['experiment_info']['total_requests']}")
        print(f"Unique algorithms expected: 20")
        print(f"Unique algorithms used: {len(report['algorithm_usage'])}")

        print("\n📊 ALGORITHM DISTRIBUTION:")
        print("-" * 80)
        for algo, count in sorted(report['expected_algorithm_distribution'].items()):
            percentage = (count / report['experiment_info']['total_requests']) * 100
            print(f"  {algo}: {count} requests ({percentage:.1f}%)")

        print("\n✅ Experiment v7.0 completed successfully!")


if __name__ == "__main__":
    main()