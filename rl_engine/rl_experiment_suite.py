import requests
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import statistics
import random
import json


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
        """Generate realistic feedback with algorithm-specific behavior"""

        category = scenario.get('category', 'general')
        security_level = scenario['context'].get('security_level', 'moderate')
        has_qkd = 'QKD' in scenario['context'].get('dst_props', {}).get('hardware', [])

        # Determine primary algorithm from proposed
        primary_algo = 'CLASSICAL'
        if proposed_algos:
            first_algo = proposed_algos[0]
            if 'QKD' in first_algo or 'BB84' in first_algo or 'E91' in first_algo or 'CV' in first_algo or 'MDI' in first_algo:
                primary_algo = 'QKD'
            elif 'KYBER' in first_algo or 'DILITHIUM' in first_algo or 'FALCON' in first_algo or 'NTRU' in first_algo or 'SABER' in first_algo:
                primary_algo = 'PQC'
            elif 'HYBRID' in first_algo:
                primary_algo = 'HYBRID'
            elif 'AES' in first_algo or 'RSA' in first_algo or 'ECC' in first_algo:
                primary_algo = 'CLASSICAL'

        # Success rates based on algorithm and context
        success_rates = {
            'QKD': {
                'with_qkd': {'ultra': 0.98, 'very_high': 0.96, 'high': 0.94, 'moderate': 0.90, 'low': 0.85},
                'without_qkd': {'ultra': 0.50, 'very_high': 0.45, 'high': 0.40, 'moderate': 0.35, 'low': 0.30}
            },
            'PQC': {
                'with_qkd': {'ultra': 0.95, 'very_high': 0.93, 'high': 0.91, 'moderate': 0.88, 'low': 0.85},
                'without_qkd': {'ultra': 0.94, 'very_high': 0.92, 'high': 0.90, 'moderate': 0.87, 'low': 0.84}
            },
            'HYBRID': {
                'with_qkd': {'ultra': 0.93, 'very_high': 0.91, 'high': 0.89, 'moderate': 0.86, 'low': 0.83},
                'without_qkd': {'ultra': 0.92, 'very_high': 0.90, 'high': 0.88, 'moderate': 0.85, 'low': 0.82}
            },
            'CLASSICAL': {
                'with_qkd': {'ultra': 0.75, 'very_high': 0.78, 'high': 0.82, 'moderate': 0.88, 'low': 0.92},
                'without_qkd': {'ultra': 0.70, 'very_high': 0.75, 'high': 0.80, 'moderate': 0.86, 'low': 0.90}
            }
        }

        qkd_key = 'with_qkd' if has_qkd else 'without_qkd'
        base_success_rate = success_rates[primary_algo][qkd_key].get(security_level, 0.85)

        # Adjust for stress scenarios
        if category == 'stress_test':
            base_success_rate *= 0.75
        elif category == 'low_resources':
            base_success_rate *= 0.70
        elif category == 'network_conditions':
            network_latency = scenario['context'].get('network_latency', 0)
            if network_latency > 200:
                base_success_rate *= 0.85

        success = random.random() < base_success_rate

        # Latency based on algorithm type
        latency_ranges = {
            'QKD': (45, 85),
            'PQC': (28, 55),
            'HYBRID': (32, 65),
            'CLASSICAL': (18, 40)
        }

        latency_range = latency_ranges.get(primary_algo, (20, 50))

        if success:
            latency = random.uniform(latency_range[0], latency_range[1])
        else:
            latency = random.uniform(latency_range[1], latency_range[1] * 2.5)

        # Add network latency
        network_latency = scenario['context'].get('network_latency', 0)
        if network_latency:
            latency += network_latency * 0.25

        # Resource usage
        system_load = scenario['context'].get('system_load', 0.5)
        resource_base = {
            'QKD': 0.78,
            'PQC': 0.58,
            'HYBRID': 0.68,
            'CLASSICAL': 0.38
        }

        resource_usage = resource_base.get(primary_algo, 0.5)
        resource_usage = min(resource_usage + (system_load * 0.15), 0.95)

        return {
            'success': success,
            'latency': round(latency, 2),
            'resource_usage': round(resource_usage, 2)
        }

    def run_scenario(self, scenario: Dict):
        """Execute a test scenario"""
        print(f"  → {scenario['name']}")

        result = self.send_request(scenario['context'])

        proposed_algos = result.get('payload', {}).get('proposed', [])
        feedback = self.generate_dynamic_feedback(scenario, proposed_algos)

        time.sleep(0.3)  # 300ms delay

        self.send_feedback(
            scenario['context']['request_id'],
            feedback['success'],
            feedback['latency'],
            feedback['resource_usage']
        )

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

        status = "✓ SUCCESS" if feedback['success'] else "✗ FAILED"
        print(f"    {status} | Latency: {feedback['latency']:.2f}ms | Resource: {feedback['resource_usage']:.2f}")

        return result_data

    def run_experiment(self, scenarios: List[Dict], episodes: int = 20,
                       iterations_per_episode: int = 10):
        """Execute complete experiment"""
        print("=" * 80)
        print("RL ENGINE - ENHANCED EXPERIMENT v3.0")
        print("=" * 80)
        print(f"Episodes: {episodes}")
        print(f"Iterations per episode: {iterations_per_episode}")
        print(f"Unique scenarios: {len(scenarios)}")
        print(f"Total requests: {episodes * iterations_per_episode * len(scenarios)}")
        print(f"Estimated time: ~{(episodes * iterations_per_episode * len(scenarios) * 0.35 / 60):.1f} minutes")

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
                    print("\\n  ⏸  Pausing between iterations...")
                    time.sleep(1.5)

            episode_result = self.end_episode()
            episode_elapsed = time.time() - episode_start

            metrics = self.get_metrics()
            metrics['episode'] = episode
            metrics['elapsed_time'] = episode_elapsed
            self.metrics_history.append(metrics)

            print(f"\\n  ✅ Episode {episode} completed in {episode_elapsed:.2f}s")
            print(f"  📊 Requests in this episode: {iterations_per_episode * len(scenarios)}")

            if episode < episodes:
                print(f"  ⏸  Pausing between episodes...")
                time.sleep(2.5)

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
                'version': '3.0'
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

    def save_results(self, report: Dict, prefix: str = "rl_experiment_v3"):
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

        metrics_csv = f"{prefix}_{timestamp}_metrics.csv"
        with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
            if report['metrics_history']:
                fieldnames = ['episode', 'elapsed_time']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(report['metrics_history'])
        print(f"✅ Metrics CSV: {metrics_csv}")

        txt_file = f"{prefix}_{timestamp}_summary.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\\n")
            f.write("RL ENGINE - ENHANCED EXPERIMENT v3.0\\n")
            f.write("=" * 80 + "\\n\\n")

            f.write("EXPERIMENT INFORMATION\\n")
            f.write("-" * 80 + "\\n")
            for key, value in report['experiment_info'].items():
                f.write(f"{key}: {value}\\n")

            f.write("\\nPERFORMANCE METRICS\\n")
            f.write("-" * 80 + "\\n")
            for key, value in report['performance_metrics'].items():
                f.write(f"{key}: {value:.2f}\\n")

            f.write("\\nALGORITHM USAGE (TOP 15)\\n")
            f.write("-" * 80 + "\\n")
            sorted_algos = sorted(report['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for algo, count in sorted_algos[:15]:
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


# BALANCED SCENARIOS - Better distribution
BALANCED_SCENARIOS = [
    # === QUANTUM (25%) - 5 scenarios ===
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
        'name': 'QKD Very High - E91',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-veryhigh-e91-002',
            'source': 'quantum-node-B',
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
        'name': 'QKD High - CV-QKD',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-high-cv-003',
            'source': 'quantum-node-C',
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
    {
        'name': 'QKD High - MDI-QKD',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-high-mdi-004',
            'source': 'quantum-node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.80,
            'conf_score': 0.85,
            'data_sensitivity': 0.83,
            'available_resources': 0.78,
            'system_load': 0.48,
            'dst_props': {
                'hardware': ['QKD'],
                'compliance': ['GDPR']
            }
        }
    },
    {
        'name': 'QKD Very High - DECOY',
        'category': 'quantum',
        'context': {
            'request_id': 'qkd-veryhigh-decoy-005',
            'source': 'quantum-node-E',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.86,
            'conf_score': 0.90,
            'data_sensitivity': 0.88,
            'available_resources': 0.85,
            'system_load': 0.38,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['HIPAA', 'SOC2']
            }
        }
    },

    # === POST-QUANTUM (30%) - 6 scenarios ===
    {
        'name': 'PQC Ultra - KYBER',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-ultra-kyber-006',
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
            'request_id': 'pqc-veryhigh-dilithium-007',
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
        'name': 'PQC Very High - NTRU',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-veryhigh-ntru-008',
            'source': 'pqc-node-C',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.83,
            'conf_score': 0.86,
            'data_sensitivity': 0.85,
            'available_resources': 0.72,
            'system_load': 0.52,
            'dst_props': {
                'hardware': [],
                'compliance': ['ISO27001']
            }
        }
    },
    {
        'name': 'PQC High - SABER',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-high-saber-009',
            'source': 'pqc-node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.78,
            'conf_score': 0.82,
            'data_sensitivity': 0.80,
            'available_resources': 0.68,
            'system_load': 0.58,
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
            'request_id': 'pqc-ultra-falcon-010',
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
    {
        'name': 'PQC High - SPHINCS',
        'category': 'post_quantum',
        'context': {
            'request_id': 'pqc-high-sphincs-011',
            'source': 'pqc-node-F',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.76,
            'conf_score': 0.80,
            'data_sensitivity': 0.78,
            'available_resources': 0.65,
            'system_load': 0.60,
            'dst_props': {
                'hardware': [],
                'compliance': ['PCI-DSS']
            }
        }
    },

    # === HYBRID (20%) - 4 scenarios ===
    {
        'name': 'Hybrid Very High - RSA+PQC',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-veryhigh-rsa-012',
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
        'name': 'Hybrid High - ECC+PQC',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-high-ecc-013',
            'source': 'hybrid-node-B',
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
        'name': 'Hybrid Very High - Mixed',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-veryhigh-mixed-014',
            'source': 'hybrid-node-C',
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
    {
        'name': 'Hybrid High - Transition',
        'category': 'hybrid',
        'context': {
            'request_id': 'hybrid-high-trans-015',
            'source': 'hybrid-node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.70,
            'conf_score': 0.74,
            'data_sensitivity': 0.72,
            'available_resources': 0.68,
            'system_load': 0.58,
            'dst_props': {
                'hardware': [],
                'compliance': ['PCI-DSS']
            }
        }
    },

    # === CLASSICAL (15%) - 3 scenarios ===
    {
        'name': 'Classical High - AES-256',
        'category': 'classical',
        'context': {
            'request_id': 'classical-high-aes256-016',
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
        'name': 'Classical Moderate - RSA-4096',
        'category': 'classical',
        'context': {
            'request_id': 'classical-mod-rsa-017',
            'source': 'classical-node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'moderate',
            'risk_score': 0.50,
            'conf_score': 0.55,
            'data_sensitivity': 0.53,
            'available_resources': 0.55,
            'system_load': 0.65,
            'dst_props': {
                'hardware': [],
                'compliance': []
            }
        }
    },
    {
        'name': 'Classical High - ECC-521',
        'category': 'classical',
        'context': {
            'request_id': 'classical-high-ecc-018',
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

    # === STRESS & NETWORK (10%) - 2 scenarios ===
    {
        'name': 'Stress - High Load + QKD',
        'category': 'stress_test',
        'context': {
            'request_id': 'stress-highload-019',
            'source': 'stress-node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.88,
            'conf_score': 0.85,
            'system_load': 0.88,
            'available_resources': 0.30,
            'dst_props': {
                'hardware': ['QKD']
            }
        }
    },
    {
        'name': 'Network - High Latency',
        'category': 'network_conditions',
        'context': {
            'request_id': 'network-highlatency-020',
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
    }
]


def main():
    print("\\n" + "=" * 80)
    print("RL ENGINE - ENHANCED EXPERIMENT v3.0")
    print("=" * 80)

    experiment = ImprovedRLExperiment(base_url="http://localhost:9009")

    report = experiment.run_experiment(
        scenarios=BALANCED_SCENARIOS,
        episodes=20,
        iterations_per_episode=10
    )

    if report:
        files = experiment.save_results(report, prefix="rl_experiment_v3")

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

        print("\\n✅ Experiment v3.0 completed successfully!")


if __name__ == "__main__":
    main()