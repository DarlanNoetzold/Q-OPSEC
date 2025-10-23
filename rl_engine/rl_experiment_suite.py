import requests
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import statistics
import random
import json

class ForcedBalancedExperiment:
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
        """Generate feedback that REWARDS the expected algorithm"""

        # Check if expected algorithm was proposed
        algo_match = any(expected_algo in algo for algo in proposed_algos)

        # Base success rate
        if algo_match:
            # REWARD: High success if correct algorithm chosen
            base_success = 0.95
            latency_multiplier = 1.0
        else:
            # PENALTY: Lower success if wrong algorithm
            base_success = 0.60
            latency_multiplier = 1.8

        # Add some randomness
        success = random.random() < base_success

        # Algorithm-specific latencies
        latency_map = {
            'BB84': (50, 70),
            'E91': (52, 72),
            'CV-QKD': (48, 68),
            'MDI-QKD': (55, 75),
            'DECOY': (53, 73),
            'KYBER': (30, 45),
            'DILITHIUM': (32, 47),
            'NTRU': (28, 43),
            'SABER': (29, 44),
            'FALCON': (31, 46),
            'SPHINCS': (35, 50),
            'RSA': (25, 40),
            'ECC': (22, 38),
            'AES': (18, 35),
            'HYBRID': (35, 55)
        }

        # Find latency range
        latency_range = (30, 50)  # default
        for key, value in latency_map.items():
            if key in expected_algo:
                latency_range = value
                break

        if success:
            latency = random.uniform(latency_range[0], latency_range[1]) * latency_multiplier
        else:
            latency = random.uniform(latency_range[1], latency_range[1] * 2.0)

        # Resource usage
        resource_map = {
            'QKD': 0.75,
            'PQC': 0.55,
            'HYBRID': 0.65,
            'CLASSICAL': 0.35
        }

        resource_usage = 0.50
        for key, value in resource_map.items():
            if key in expected_algo:
                resource_usage = value
                break

        return {
            'success': success,
            'latency': round(latency, 2),
            'resource_usage': round(resource_usage + random.uniform(-0.05, 0.05), 2)
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

        time.sleep(0.2)  # 200ms delay

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

    def run_experiment(self, scenarios: List[Dict], episodes: int = 25,
                       iterations_per_episode: int = 8):
        """Execute complete experiment"""
        print("=" * 80)
        print("RL ENGINE - FORCED BALANCED EXPERIMENT v4.0")
        print("=" * 80)
        print(f"Episodes: {episodes}")
        print(f"Iterations per episode: {iterations_per_episode}")
        print(f"Unique scenarios: {len(scenarios)}")
        print(f"Total requests: {episodes * iterations_per_episode * len(scenarios)}")
        print(f"Estimated time: ~{(episodes * iterations_per_episode * len(scenarios) * 0.25 / 60):.1f} minutes")

        if not self.health_check():
            print("\\n❌ ERROR: RL Engine is not responding!")
            return

        print("\\n✅ RL Engine is online!")

        self.enable_training()
        print("✅ Training mode enabled")

        experiment_start = time.time()

        for episode in range(1, episodes + 1):
            print(f"\\n{'=' * 80}")
            print(f"EPISODE {episode}/{episodes}")
            print(f"{'=' * 80}")

            episode_start = time.time()

            for iteration in range(iterations_per_episode):
                print(f"\\n[Iteration {iteration + 1}/{iterations_per_episode}]")

                shuffled_scenarios = random.sample(scenarios, len(scenarios))

                for idx, scenario in enumerate(shuffled_scenarios, 1):
                    print(f"  [{idx}/{len(scenarios)}]", end=" ")
                    self.run_scenario(scenario)

                if iteration < iterations_per_episode - 1:
                    time.sleep(1.0)

            episode_result = self.end_episode()
            episode_elapsed = time.time() - episode_start

            metrics = self.get_metrics()
            metrics['episode'] = episode
            metrics['elapsed_time'] = episode_elapsed
            self.metrics_history.append(metrics)

            print(f"\\n  ✅ Episode {episode} completed in {episode_elapsed:.2f}s")

            if episode < episodes:
                time.sleep(1.5)

        experiment_elapsed = time.time() - experiment_start

        print(f"\\n{'=' * 80}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'=' * 80}")
        print(f"Total time: {experiment_elapsed:.2f}s ({experiment_elapsed / 60:.2f} minutes)")
        print(f"Total requests: {len(self.results)}")

        self.disable_training()
        print("✅ Training mode disabled")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate complete report"""
        print("\\n📊 Generating report...")

        algorithm_usage = {}
        for result in self.results:
            for algo in result['proposed_algorithms']:
                algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1

        expected_algo_usage = {}
        for result in self.results:
            exp = result['expected_algorithm']
            expected_algo_usage[exp] = expected_algo_usage.get(exp, 0) + 1

        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['feedback_success'])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        latencies = [r['feedback_latency'] for r in self.results]
        avg_latency = statistics.mean(latencies) if latencies else 0

        report = {
            'experiment_info': {
                'total_requests': total_requests,
                'total_episodes': len(self.metrics_history),
                'timestamp': datetime.now().isoformat(),
                'version': '4.0'
            },
            'performance_metrics': {
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0
            },
            'algorithm_usage': algorithm_usage,
            'expected_algorithm_distribution': expected_algo_usage,
            'metrics_history': self.metrics_history,
            'raw_results': self.results
        }

        return report

    def save_results(self, report: Dict, prefix: str = "rl_experiment_v4"):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\\n✅ JSON Report: {json_file}")

        csv_file = f"{prefix}_{timestamp}_details.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if report['raw_results']:
                writer = csv.DictWriter(f, fieldnames=report['raw_results'][0].keys())
                writer.writeheader()
                writer.writerows(report['raw_results'])
        print(f"✅ Details CSV: {csv_file}")

        txt_file = f"{prefix}_{timestamp}_summary.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\\n")
            f.write("RL ENGINE - FORCED BALANCED EXPERIMENT v4.0\\n")
            f.write("=" * 80 + "\\n\\n")

            f.write("PERFORMANCE METRICS\\n")
            f.write("-" * 80 + "\\n")
            for key, value in report['performance_metrics'].items():
                f.write(f"{key}: {value:.2f}\\n")

            f.write("\\nEXPECTED ALGORITHM DISTRIBUTION\\n")
            f.write("-" * 80 + "\\n")
            for algo, count in sorted(report['expected_algorithm_distribution'].items()):
                percentage = (count / report['experiment_info']['total_requests']) * 100
                f.write(f"{algo}: {count} times ({percentage:.1f}%)\\n")

            f.write("\\nACTUAL ALGORITHM USAGE (TOP 20)\\n")
            f.write("-" * 80 + "\\n")
            sorted_algos = sorted(report['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for algo, count in sorted_algos[:20]:
                percentage = (count / report['experiment_info']['total_requests']) * 100
                f.write(f"{algo}: {count} times ({percentage:.1f}%)\\n")

        print(f"✅ Summary TXT: {txt_file}")

        return {
            'json': json_file,
            'csv': csv_file,
            'summary': txt_file
        }


# FORCED BALANCED SCENARIOS - Each algorithm gets equal representation
FORCED_SCENARIOS = [
    # === QKD BB84 (6.25%) ===
    {
        'name': 'QKD BB84 - Ultra Security',
        'category': 'quantum_bb84',
        'expected_algorithm': 'QKD_BB84',
        'context': {
            'request_id': 'qkd-bb84-001',
            'source': 'quantum-bb84-node',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.95,
            'conf_score': 0.98,
            'data_sensitivity': 0.98,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['FIPS-140-3']
            }
        }
    },

    # === QKD E91 (6.25%) ===
    {
        'name': 'QKD E91 - Very High Security',
        'category': 'quantum_e91',
        'expected_algorithm': 'QKD_E91',
        'context': {
            'request_id': 'qkd-e91-002',
            'source': 'quantum-e91-node',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.90,
            'conf_score': 0.93,
            'data_sensitivity': 0.92,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['ISO27001']
            }
        }
    },

    # === QKD CV-QKD (6.25%) ===
    {
        'name': 'QKD CV-QKD - High Security',
        'category': 'quantum_cv',
        'expected_algorithm': 'QKD_CV-QKD',
        'context': {
            'request_id': 'qkd-cv-003',
            'source': 'quantum-cv-node',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.85,
            'conf_score': 0.88,
            'data_sensitivity': 0.87,
            'dst_props': {
                'hardware': ['QKD'],
                'compliance': ['GDPR']
            }
        }
    },

    # === QKD MDI-QKD (6.25%) ===
    {
        'name': 'QKD MDI-QKD - High Security',
        'category': 'quantum_mdi',
        'expected_algorithm': 'QKD_MDI-QKD',
        'context': {
            'request_id': 'qkd-mdi-004',
            'source': 'quantum-mdi-node',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.83,
            'conf_score': 0.86,
            'data_sensitivity': 0.85,
            'dst_props': {
                'hardware': ['QKD'],
                'compliance': ['SOC2']
            }
        }
    },

    # === PQC KYBER (6.25%) ===
    {
        'name': 'PQC KYBER - Ultra Security',
        'category': 'pqc_kyber',
        'expected_algorithm': 'PQC_KYBER',
        'context': {
            'request_id': 'pqc-kyber-005',
            'source': 'pqc-kyber-node',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.93,
            'conf_score': 0.96,
            'data_sensitivity': 0.95,
            'dst_props': {
                'hardware': [],
                'compliance': ['NIST-PQC']
            }
        }
    },

    # === PQC DILITHIUM (6.25%) ===
    {
        'name': 'PQC DILITHIUM - Very High Security',
        'category': 'pqc_dilithium',
        'expected_algorithm': 'PQC_DILITHIUM',
        'context': {
            'request_id': 'pqc-dilithium-006',
            'source': 'pqc-dilithium-node',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.88,
            'conf_score': 0.91,
            'data_sensitivity': 0.90,
            'dst_props': {
                'hardware': [],
                'compliance': ['FIPS-140-3']
            }
        }
    },

    # === PQC NTRU (6.25%) ===
    {
        'name': 'PQC NTRU - Very High Security',
        'category': 'pqc_ntru',
        'expected_algorithm': 'PQC_NTRU',
        'context': {
            'request_id': 'pqc-ntru-007',
            'source': 'pqc-ntru-node',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.86,
            'conf_score': 0.89,
            'data_sensitivity': 0.88,
            'dst_props': {
                'hardware': [],
                'compliance': ['GDPR']
            }
        }
    },

    # === PQC SABER (6.25%) ===
    {
        'name': 'PQC SABER - High Security',
        'category': 'pqc_saber',
        'expected_algorithm': 'PQC_SABER',
        'context': {
            'request_id': 'pqc-saber-008',
            'source': 'pqc-saber-node',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.80,
            'conf_score': 0.84,
            'data_sensitivity': 0.82,
            'dst_props': {
                'hardware': [],
                'compliance': ['ISO27001']
            }
        }
    },

    # === PQC FALCON (6.25%) ===
    {
        'name': 'PQC FALCON - Ultra Security',
        'category': 'pqc_falcon',
        'expected_algorithm': 'PQC_FALCON',
        'context': {
            'request_id': 'pqc-falcon-009',
            'source': 'pqc-falcon-node',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.94,
            'conf_score': 0.97,
            'data_sensitivity': 0.96,
            'dst_props': {
                'hardware': [],
                'compliance': ['NIST-PQC', 'FIPS-140-3']
            }
        }
    },

    # === HYBRID RSA+PQC (6.25%) ===
    {
        'name': 'HYBRID RSA+PQC - Very High',
        'category': 'hybrid_rsa',
        'expected_algorithm': 'HYBRID_RSA_PQC',
        'context': {
            'request_id': 'hybrid-rsa-010',
            'source': 'hybrid-rsa-node',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.82,
            'conf_score': 0.85,
            'data_sensitivity': 0.84,
            'dst_props': {
                'hardware': [],
                'compliance': ['SOC2']
            }
        }
    },

    # === HYBRID ECC+PQC (6.25%) ===
    {
        'name': 'HYBRID ECC+PQC - High',
        'category': 'hybrid_ecc',
        'expected_algorithm': 'HYBRID_ECC_PQC',
        'context': {
            'request_id': 'hybrid-ecc-011',
            'source': 'hybrid-ecc-node',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.75,
            'conf_score': 0.79,
            'data_sensitivity': 0.77,
            'dst_props': {
                'hardware': [],
                'compliance': ['GDPR']
            }
        }
    },

    # === RSA 4096 (6.25%) ===
    {
        'name': 'RSA 4096 - High Security',
        'category': 'classical_rsa',
        'expected_algorithm': 'RSA_4096',
        'context': {
            'request_id': 'rsa-4096-012',
            'source': 'classical-rsa-node',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.70,
            'conf_score': 0.74,
            'data_sensitivity': 0.72,
            'dst_props': {
                'hardware': [],
                'compliance': ['FIPS-140-2']
            }
        }
    },

    # === ECC 521 (6.25%) ===
    {
        'name': 'ECC 521 - High Security',
        'category': 'classical_ecc',
        'expected_algorithm': 'ECC_521',
        'context': {
            'request_id': 'ecc-521-013',
            'source': 'classical-ecc-node',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.72,
            'conf_score': 0.76,
            'data_sensitivity': 0.74,
            'dst_props': {
                'hardware': [],
                'compliance': ['PCI-DSS']
            }
        }
    },

    # === AES 256 GCM (6.25%) ===
    {
        'name': 'AES 256 GCM - Moderate Security',
        'category': 'classical_aes',
        'expected_algorithm': 'AES_256_GCM',
        'context': {
            'request_id': 'aes-256-014',
            'source': 'classical-aes-node',
            'destination': 'http://localhost:9000',
            'security_level': 'moderate',
            'risk_score': 0.55,
            'conf_score': 0.60,
            'data_sensitivity': 0.58,
            'dst_props': {
                'hardware': [],
                'compliance': []
            }
        }
    },

    # === AES 192 (6.25%) ===
    {
        'name': 'AES 192 - Moderate Security',
        'category': 'classical_aes192',
        'expected_algorithm': 'AES_192',
        'context': {
            'request_id': 'aes-192-015',
            'source': 'classical-aes192-node',
            'destination': 'http://localhost:9000',
            'security_level': 'moderate',
            'risk_score': 0.50,
            'conf_score': 0.55,
            'data_sensitivity': 0.53,
            'dst_props': {
                'hardware': [],
                'compliance': []
            }
        }
    },

    # === FALLBACK AES (6.25%) ===
    {
        'name': 'FALLBACK AES - Low Resources',
        'category': 'fallback',
        'expected_algorithm': 'FALLBACK_AES',
        'context': {
            'request_id': 'fallback-aes-016',
            'source': 'fallback-node',
            'destination': 'http://localhost:9000',
            'security_level': 'moderate',
            'risk_score': 0.60,
            'conf_score': 0.65,
            'available_resources': 0.25,
            'system_load': 0.90,
            'dst_props': {
                'hardware': [],
                'compliance': []
            }
        }
    }
]


def main():
    print("\\n" + "=" * 80)
    print("RL ENGINE - FORCED BALANCED EXPERIMENT v4.0")
    print("=" * 80)

    experiment = ForcedBalancedExperiment(base_url="http://localhost:9009")

    report = experiment.run_experiment(
        scenarios=FORCED_SCENARIOS,
        episodes=25,
        iterations_per_episode=8
    )

    if report:
        files = experiment.save_results(report, prefix="rl_experiment_v4")

        print("\\n" + "=" * 80)
        print("GENERATED FILES:")
        print("=" * 80)
        for file_type, filename in files.items():
            print(f"  {file_type}: {filename}")

        print("\\n" + "=" * 80)
        print("QUICK SUMMARY:")
        print("=" * 80)
        print(f"Success rate: {report['performance_metrics']['success_rate']:.2f}%")
        print(f"Average latency: {report['performance_metrics']['avg_latency_ms']:.2f}ms")
        print(f"Total requests: {report['experiment_info']['total_requests']}")
        print(f"Unique algorithms used: {len(report['algorithm_usage'])}")

        print("\\n✅ Experiment v4.0 completed successfully!")


if __name__ == "__main__":
    main()