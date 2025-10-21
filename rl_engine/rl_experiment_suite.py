import requests
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import statistics
import random


class ImprovedRLExperiment:
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

    def generate_dynamic_feedback(self, scenario: Dict, proposed_algos: List[str]) -> Dict:
        """Generate more realistic feedback based on scenario and algorithms"""

        # Base success rate by category
        category = scenario.get('category', 'general')
        base_success_rates = {
            'quantum': 0.95,
            'post_quantum': 0.92,
            'hybrid': 0.90,
            'classical': 0.88,
            'stress_test': 0.75,
            'network_conditions': 0.82,
            'temporal': 0.85,
            'low_resources': 0.70
        }

        base_rate = base_success_rates.get(category, 0.85)

        # Adjust based on security level
        security_level = scenario['context'].get('security_level', 'moderate')
        security_modifiers = {
            'ultra': 0.98,
            'very_high': 0.95,
            'high': 0.92,
            'moderate': 0.88,
            'low': 0.85
        }

        success_rate = base_rate * security_modifiers.get(security_level, 0.85)

        # Penalize if QKD is needed but not available
        has_qkd = 'QKD' in scenario['context'].get('dst_props', {}).get('hardware', [])
        needs_high_security = security_level in ['ultra', 'very_high']

        if needs_high_security and not has_qkd:
            success_rate *= 0.85

        # Determine success
        success = random.random() < success_rate

        # Latency based on algorithm and conditions
        base_latencies = {
            'QKD': (40, 80),
            'PQC': (25, 50),
            'HYBRID': (30, 60),
            'AES': (15, 35),
            'RSA': (20, 45),
            'ECC': (18, 40),
            'FALLBACK': (10, 25)
        }

        # Determine latency based on first proposed algorithm
        algo_type = 'AES'  # default
        if proposed_algos:
            first_algo = proposed_algos[0]
            for key in base_latencies.keys():
                if key in first_algo:
                    algo_type = key
                    break

        latency_range = base_latencies.get(algo_type, (20, 50))

        if success:
            latency = random.uniform(latency_range[0], latency_range[1])
        else:
            # Failures have higher latency
            latency = random.uniform(latency_range[1], latency_range[1] * 3)

        # Add network latency if specified
        network_latency = scenario['context'].get('network_latency', 0)
        if network_latency:
            latency += network_latency * 0.3  # 30% of network latency

        # Resource usage based on algorithm and system load
        system_load = scenario['context'].get('system_load', 0.5)

        resource_base = {
            'QKD': 0.75,
            'PQC': 0.55,
            'HYBRID': 0.65,
            'AES': 0.35,
            'RSA': 0.45,
            'ECC': 0.40,
            'FALLBACK': 0.25
        }

        resource_usage = resource_base.get(algo_type, 0.5)
        resource_usage = min(resource_usage + (system_load * 0.2), 0.95)

        return {
            'success': success,
            'latency': round(latency, 2),
            'resource_usage': round(resource_usage, 2)
        }

    def run_scenario(self, scenario: Dict):
        """Execute a test scenario"""
        print(f"  → {scenario['name']}")

        result = self.send_request(scenario['context'])

        # Generate realistic feedback
        proposed_algos = result.get('payload', {}).get('proposed', [])
        feedback = self.generate_dynamic_feedback(scenario, proposed_algos)

        # Slower execution - wait between request and feedback
        time.sleep(0.2)  # 200ms delay

        self.send_feedback(
            scenario['context']['request_id'],
            feedback['success'],
            feedback['latency'],
            feedback['resource_usage']
        )

        # Store result
        result_data = {
            'scenario_name': scenario['name'],
            'scenario_category': scenario.get('category', 'general'),
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
            'timestamp': result.get('timestamp'),
            'system_load': scenario['context'].get('system_load'),
            'available_resources': scenario['context'].get('available_resources'),
            'network_latency': scenario['context'].get('network_latency')
        }

        self.results.append(result_data)

        # Show feedback result
        status = "✓ SUCCESS" if feedback['success'] else "✗ FAILED"
        print(f"    {status} | Latency: {feedback['latency']:.2f}ms | Resource: {feedback['resource_usage']:.2f}")

        return result_data

    def run_experiment(self, scenarios: List[Dict], episodes: int = 15,
                       iterations_per_episode: int = 8):
        """Execute complete experiment"""
        print("=" * 80)
        print("RL ENGINE - ENHANCED EXPERIMENT v2.0")
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

                # Shuffle scenarios for greater variation
                shuffled_scenarios = random.sample(scenarios, len(scenarios))

                for idx, scenario in enumerate(shuffled_scenarios, 1):
                    print(f"  [{idx}/{len(scenarios)}]", end=" ")
                    self.run_scenario(scenario)

                # Pause between iterations
                if iteration < iterations_per_episode - 1:
                    print("\\n  ⏸  Pausing between iterations...")
                    time.sleep(1.0)

            episode_result = self.end_episode()
            episode_elapsed = time.time() - episode_start

            metrics = self.get_metrics()
            metrics['episode'] = episode
            metrics['elapsed_time'] = episode_elapsed
            self.metrics_history.append(metrics)

            print(f"\\n  ✅ Episode {episode} completed in {episode_elapsed:.2f}s")
            print(f"  📊 Requests in this episode: {iterations_per_episode * len(scenarios)}")

            # Longer pause between episodes
            if episode < episodes:
                print(f"  ⏸  Pausing between episodes...")
                time.sleep(2.0)

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

        by_security_level = {}
        for result in self.results:
            level = result['security_level']
            if level not in by_security_level:
                by_security_level[level] = []
            by_security_level[level].append(result)

        by_category = {}
        for result in self.results:
            cat = result['scenario_category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)

        with_qkd = [r for r in self.results if r['has_qkd']]
        without_qkd = [r for r in self.results if not r['has_qkd']]

        algorithm_usage = {}
        for result in self.results:
            for algo in result['proposed_algorithms']:
                algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1

        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['feedback_success'])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        latencies = [r['feedback_latency'] for r in self.results]
        avg_latency = statistics.mean(latencies) if latencies else 0
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        response_times = [r['response_time'] for r in self.results]
        avg_response_time = statistics.mean(response_times) if response_times else 0

        report = {
            'experiment_info': {
                'total_requests': total_requests,
                'total_episodes': len(self.metrics_history),
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            },
            'performance_metrics': {
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
                'std_latency_ms': std_latency,
                'avg_response_time_s': avg_response_time,
                'min_latency_ms': min(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0
            },
            'algorithm_usage': algorithm_usage,
            'by_security_level': {
                level: {
                    'count': len(results),
                    'success_rate': sum(1 for r in results if r['feedback_success']) / len(results) * 100,
                    'avg_latency': statistics.mean([r['feedback_latency'] for r in results])
                }
                for level, results in by_security_level.items()
            },
            'by_category': {
                cat: {
                    'count': len(results),
                    'success_rate': sum(1 for r in results if r['feedback_success']) / len(results) * 100,
                    'avg_latency': statistics.mean([r['feedback_latency'] for r in results])
                }
                for cat, results in by_category.items()
            },
            'qkd_analysis': {
                'with_qkd': {
                    'count': len(with_qkd),
                    'success_rate': sum(1 for r in with_qkd if r['feedback_success']) / len(
                        with_qkd) * 100 if with_qkd else 0,
                    'avg_latency': statistics.mean([r['feedback_latency'] for r in with_qkd]) if with_qkd else 0
                },
                'without_qkd': {
                    'count': len(without_qkd),
                    'success_rate': sum(1 for r in without_qkd if r['feedback_success']) / len(
                        without_qkd) * 100 if without_qkd else 0,
                    'avg_latency': statistics.mean([r['feedback_latency'] for r in without_qkd]) if without_qkd else 0
                }
            },
            'metrics_history': self.metrics_history,
            'raw_results': self.results
        }

        return report

    def save_results(self, report: Dict, prefix: str = "rl_experiment_v2"):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_file = f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\\n✅ JSON Report: {json_file}")

        # Detailed CSV
        csv_file = f"{prefix}_{timestamp}_details.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if report['raw_results']:
                writer = csv.DictWriter(f, fieldnames=report['raw_results'][0].keys())
                writer.writeheader()
                writer.writerows(report['raw_results'])
        print(f"✅ Details CSV: {csv_file}")

        # Metrics CSV
        metrics_csv = f"{prefix}_{timestamp}_metrics.csv"
        with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
            if report['metrics_history']:
                fieldnames = ['episode', 'elapsed_time']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(report['metrics_history'])
        print(f"✅ Metrics CSV: {metrics_csv}")

        # Enhanced text summary
        txt_file = f"{prefix}_{timestamp}_summary.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\\n")
            f.write("RL ENGINE - ENHANCED EXPERIMENT v2.0\\n")
            f.write("=" * 80 + "\\n\\n")

            f.write("EXPERIMENT INFORMATION\\n")
            f.write("-" * 80 + "\\n")
            for key, value in report['experiment_info'].items():
                f.write(f"{key}: {value}\\n")

            f.write("\\nPERFORMANCE METRICS\\n")
            f.write("-" * 80 + "\\n")
            for key, value in report['performance_metrics'].items():
                f.write(f"{key}: {value:.2f}\\n")

            f.write("\\nALGORITHM USAGE (TOP 10)\\n")
            f.write("-" * 80 + "\\n")
            sorted_algos = sorted(report['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for algo, count in sorted_algos[:10]:
                percentage = (count / report['experiment_info']['total_requests']) * 100
                f.write(f"{algo}: {count} times ({percentage:.1f}%)\\n")

            f.write("\\nANALYSIS BY CATEGORY\\n")
            f.write("-" * 80 + "\\n")
            for cat, data in report['by_category'].items():
                f.write(f"\\n{cat.upper()}:\\n")
                f.write(f"  Requests: {data['count']}\\n")
                f.write(f"  Success rate: {data['success_rate']:.2f}%\\n")
                f.write(f"  Average latency: {data['avg_latency']:.2f}ms\\n")

            f.write("\\nANALYSIS BY SECURITY LEVEL\\n")
            f.write("-" * 80 + "\\n")
            for level, data in report['by_security_level'].items():
                f.write(f"\\n{level.upper()}:\\n")
                f.write(f"  Requests: {data['count']}\\n")
                f.write(f"  Success rate: {data['success_rate']:.2f}%\\n")
                f.write(f"  Average latency: {data['avg_latency']:.2f}ms\\n")

            f.write("\\nQKD COMPARISON\\n")
            f.write("-" * 80 + "\\n")
            f.write("With QKD:\\n")
            f.write(f"  Requests: {report['qkd_analysis']['with_qkd']['count']}\\n")
            f.write(f"  Success rate: {report['qkd_analysis']['with_qkd']['success_rate']:.2f}%\\n")
            f.write(f"  Average latency: {report['qkd_analysis']['with_qkd']['avg_latency']:.2f}ms\\n")
            f.write("\\nWithout QKD:\\n")
            f.write(f"  Requests: {report['qkd_analysis']['without_qkd']['count']}\\n")
            f.write(f"  Success rate: {report['qkd_analysis']['without_qkd']['success_rate']:.2f}%\\n")
            f.write(f"  Average latency: {report['qkd_analysis']['without_qkd']['avg_latency']:.2f}ms\\n")

        print(f"✅ Summary TXT: {txt_file}")

        return {
            'json': json_file,
            'csv': csv_file,
            'metrics_csv': metrics_csv,
            'summary': txt_file
        }


# ENHANCED SCENARIOS - Greater diversity
IMPROVED_SCENARIOS = [
    # === QUANTUM (QKD) - 20% ===
    {
        'name': 'QKD Ultra Security - BB84',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-ultra-bb84-001',
            'source': 'quantum-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.95,
            'conf_score': 0.98,
            'data_sensitivity': 0.98,
            'service_criticality': 0.95,
            'available_resources': 0.9,
            'system_load': 0.3,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['GDPR', 'HIPAA', 'FIPS-140-3']
            }
        }
    },
    {
        'name': 'QKD High Security - E91',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-high-e91-002',
            'source': 'quantum-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.85,
            'conf_score': 0.90,
            'data_sensitivity': 0.88,
            'available_resources': 0.85,
            'system_load': 0.4,
            'dst_props': {
                'hardware': ['QKD'],
                'compliance': ['GDPR']
            }
        }
    },
    {
        'name': 'QKD Very High - CV-QKD',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-veryhigh-cv-003',
            'source': 'quantum-node-C',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.88,
            'conf_score': 0.92,
            'data_sensitivity': 0.90,
            'available_resources': 0.88,
            'system_load': 0.35,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['SOC2', 'ISO27001']
            }
        }
    },
    {
        'name': 'QKD High - MDI-QKD',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-high-mdi-004',
            'source': 'quantum-node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.82,
            'conf_score': 0.87,
            'data_sensitivity': 0.85,
            'available_resources': 0.80,
            'system_load': 0.45,
            'dst_props': {
                'hardware': ['QKD'],
                'compliance': ['PCI-DSS']
            }
        }
    },

    # === POST-QUANTUM (PQC) - 25% ===
    {
        'name': 'PQC Ultra - KYBER',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-ultra-kyber-005',
            'source': 'pqc-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.92,
            'conf_score': 0.95,
            'data_sensitivity': 0.93,
            'available_resources': 0.75,
            'system_load': 0.5,
            'dst_props': {
                'hardware': [],
                'compliance': ['NIST-PQC', 'FIPS-140-3']
            }
        }
    },
    {
        'name': 'PQC Very High - DILITHIUM',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-veryhigh-dilithium-006',
            'source': 'pqc-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.85,
            'conf_score': 0.88,
            'data_sensitivity': 0.87,
            'available_resources': 0.70,
            'system_load': 0.55,
            'dst_props': {
                'hardware': [],
                'compliance': ['GDPR', 'SOC2']
            }
        }
    },
    {
        'name': 'PQC High - NTRU',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-high-ntru-007',
            'source': 'pqc-node-C',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.80,
            'conf_score': 0.83,
            'data_sensitivity': 0.82,
            'available_resources': 0.68,
            'system_load': 0.58,
            'dst_props': {
                'hardware': [],
                'compliance': ['ISO27001']
            }
        }
    },
    {
        'name': 'PQC Very High - SABER',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-veryhigh-saber-008',
            'source': 'pqc-node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.83,
            'conf_score': 0.86,
            'data_sensitivity': 0.85,
            'available_resources': 0.72,
            'system_load': 0.52,
            'dst_props': {
                'hardware': [],
                'compliance': ['HIPAA']
            }
        }
    },
    {
        'name': 'PQC Ultra - FALCON',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-ultra-falcon-009',
            'source': 'pqc-node-E',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.94,
            'conf_score': 0.96,
            'data_sensitivity': 0.95,
            'available_resources': 0.78,
            'system_load': 0.48,
            'dst_props': {
                'hardware': [],
                'compliance': ['NIST-PQC', 'GDPR']
            }
        }
    },

    # === HYBRID - 20% ===
    {
        'name': 'Hybrid Very High - RSA+PQC',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-veryhigh-rsa-010',
            'source': 'hybrid-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.78,
            'conf_score': 0.82,
            'data_sensitivity': 0.80,
            'available_resources': 0.75,
            'system_load': 0.50,
            'dst_props': {
                'hardware': [],
                'compliance': ['FIPS-140-2', 'SOC2']
            }
        }
    },
    {
        'name': 'Hybrid Very High - ECC+PQC',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-veryhigh-ecc-011',
            'source': 'hybrid-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.76,
            'conf_score': 0.80,
            'data_sensitivity': 0.78,
            'available_resources': 0.73,
            'system_load': 0.52,
            'dst_props': {
                'hardware': [],
                'compliance': ['PCI-DSS']
            }
        }
    },
    {
        'name': 'Hybrid High - Mixed',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-high-mixed-012',
            'source': 'hybrid-node-C',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.72,
            'conf_score': 0.76,
            'data_sensitivity': 0.74,
            'available_resources': 0.70,
            'system_load': 0.55,
            'dst_props': {
                'hardware': [],
                'compliance': ['GDPR']
            }
        }
    },
    {
        'name': 'Hybrid Very High - Transition',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-veryhigh-trans-013',
            'source': 'hybrid-node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.81,
            'conf_score': 0.84,
            'data_sensitivity': 0.83,
            'available_resources': 0.76,
            'system_load': 0.48,
            'dst_props': {
                'hardware': [],
                'compliance': ['ISO27001', 'SOC2']
            }
        }
    },

    # === CLASSICAL - 15% ===
    {
        'name': 'Classical High - AES-256',
        'category': 'classical',
        'context': {
            'request_id': 'classical-high-aes256-014',
            'source': 'classical-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.65,
            'conf_score': 0.70,
            'data_sensitivity': 0.68,
            'available_resources': 0.60,
            'system_load': 0.60,
            'dst_props': {
                'hardware': [],
                'compliance': ['PCI-DSS']
            }
        }
    },
    {
        'name': 'Classical Moderate - AES-192',
        'category': 'classical',
        'context': {
            'request_id': 'classical-mod-aes192-015',
            'source': 'classical-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'moderate',
            'risk_score': 0.45,
            'conf_score': 0.50,
            'data_sensitivity': 0.48,
            'available_resources': 0.55,
            'system_load': 0.65,
            'dst_props': {
                'hardware': [],
                'compliance': []
            }
        }
    },
    {
        'name': 'Classical High - RSA-4096',
        'category': 'classical',
        'context': {
            'request_id': 'classical-high-rsa-016',
            'source': 'classical-node-C',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.68,
            'conf_score': 0.72,
            'data_sensitivity': 0.70,
            'available_resources': 0.58,
            'system_load': 0.62,
            'dst_props': {
                'hardware': [],
                'compliance': ['FIPS-140-2']
            }
        }
    },

    # === STRESS TEST - 10% ===
    {
        'name': 'Stress - Peak Attack + QKD',
        'category': 'stress_test',
        'context': {
            'request_id': 'stress-peak-qkd-017',
            'source': 'stress-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.92,
            'conf_score': 0.88,
            'is_peak_attack_time': True,
            'current_threat_level': 0.95,
            'system_load': 0.88,
            'available_resources': 0.30,
            'dst_props': {
                'hardware': ['QKD']
            }
        }
    },
    {
        'name': 'Stress - Limited Resources',
        'category': 'low_resources',
        'context': {
            'request_id': 'stress-lowres-018',
            'source': 'stress-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.75,
            'conf_score': 0.78,
            'system_load': 0.92,
            'available_resources': 0.20,
            'dst_props': {
                'hardware': []
            }
        }
    },

    # === NETWORK CONDITIONS - 10% ===
    {
        'name': 'Network - High Latency',
        'category': 'network_conditions',
        'context': {
            'request_id': 'network-highlatency-019',
            'source': 'network-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.70,
            'conf_score': 0.73,
            'network_latency': 280.0,
            'available_resources': 0.65,
            'system_load': 0.55,
            'dst_props': {
                'hardware': []
            }
        }
    },
    {
        'name': 'Network - Low Latency + QKD',
        'category': 'network_conditions',
        'context': {
            'request_id': 'network-lowlatency-020',
            'source': 'network-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.85,
            'conf_score': 0.88,
            'network_latency': 15.0,
            'available_resources': 0.85,
            'system_load': 0.35,
            'dst_props': {
                'hardware': ['QKD']
            }
        }
    }
]


def main():
    print("\\n" + "=" * 80)
    print("RL ENGINE - ENHANCED EXPERIMENT v2.0")
    print("=" * 80)

    experiment = ImprovedRLExperiment(base_url="http://localhost:9009")

    # Execute with more episodes and iterations
    report = experiment.run_experiment(
        scenarios=IMPROVED_SCENARIOS,
        episodes=15,  # Increased from 10 to 15
        iterations_per_episode=8  # Increased from 5 to 8
    )

    if report:
        files = experiment.save_results(report, prefix="rl_experiment_v2")

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
        print(f"Response time: {report['performance_metrics']['avg_response_time_s']:.4f}s")
        print(f"Total requests: {report['experiment_info']['total_requests']}")
        print(f"Unique algorithms used: {len(report['algorithm_usage'])}")

        print("\\n✅ Experiment v2.0 completed successfully!")


if __name__ == "__main__":
    main()