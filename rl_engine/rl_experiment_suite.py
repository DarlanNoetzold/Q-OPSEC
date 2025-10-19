import requests
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import statistics


class RLEngineExperiment:
    def __init__(self, base_url: str = "http://localhost:9009"):
        self.base_url = base_url
        self.results = []
        self.metrics_history = []
        self.episode_results = []

    def health_check(self) -> bool:
        """Verifica se o serviço está rodando"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def enable_training(self):
        """Ativa modo de treinamento"""
        response = requests.post(f"{self.base_url}/training/enable")
        return response.json()

    def disable_training(self):
        """Desativa modo de treinamento"""
        response = requests.post(f"{self.base_url}/training/disable")
        return response.json()

    def get_metrics(self) -> Dict:
        """Obtém métricas atuais"""
        response = requests.get(f"{self.base_url}/metrics")
        return response.json()

    def end_episode(self) -> Dict:
        """Finaliza episódio de treinamento"""
        response = requests.post(f"{self.base_url}/episode/end")
        return response.json()

    def send_request(self, context: Dict) -> Dict:
        """Envia requisição para o RL Engine"""
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
        """Envia feedback sobre resultado"""
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

    def run_scenario(self, scenario: Dict, feedback_success_rate: float = 0.9):
        """Executa um cenário de teste"""
        print(f"\n  → Executando: {scenario['name']}")

        result = self.send_request(scenario['context'])

        # Simula feedback baseado na taxa de sucesso
        import random
        success = random.random() < feedback_success_rate
        latency = random.uniform(10, 100) if success else random.uniform(100, 500)
        resource_usage = random.uniform(0.3, 0.8)

        time.sleep(0.1)  # Pequeno delay

        self.send_feedback(
            scenario['context']['request_id'],
            success,
            latency,
            resource_usage
        )

        # Armazena resultado
        result_data = {
            'scenario_name': scenario['name'],
            'scenario_category': scenario.get('category', 'general'),
            'request_id': scenario['context']['request_id'],
            'security_level': scenario['context']['security_level'],
            'risk_score': scenario['context'].get('risk_score'),
            'conf_score': scenario['context'].get('conf_score'),
            'has_qkd': 'QKD' in scenario['context'].get('dst_props', {}).get('hardware', []),
            'proposed_algorithms': result.get('payload', {}).get('proposed', []),
            'fallback_algorithms': result.get('payload', {}).get('fallback', []),
            'response_time': result.get('response_time'),
            'feedback_success': success,
            'feedback_latency': latency,
            'feedback_resource_usage': resource_usage,
            'timestamp': result.get('timestamp')
        }

        self.results.append(result_data)
        return result_data

    def run_experiment(self, scenarios: List[Dict], episodes: int = 5, 
                      iterations_per_episode: int = 10):
        """Executa experimento completo"""
        print("="*70)
        print("INICIANDO EXPERIMENTO - RL ENGINE")
        print("="*70)
        print(f"Episódios: {episodes}")
        print(f"Iterações por episódio: {iterations_per_episode}")
        print(f"Total de cenários: {len(scenarios)}")
        print(f"Total de requisições: {episodes * iterations_per_episode * len(scenarios)}")

        if not self.health_check():
            print("\n❌ ERRO: RL Engine não está respondendo!")
            return

        print("\n✅ RL Engine está online!")

        # Ativa treinamento
        self.enable_training()
        print("✅ Modo de treinamento ativado")

        for episode in range(1, episodes + 1):
            print(f"\n{'='*70}")
            print(f"EPISÓDIO {episode}/{episodes}")
            print(f"{'='*70}")

            episode_start = time.time()

            for iteration in range(iterations_per_episode):
                print(f"\nIteração {iteration + 1}/{iterations_per_episode}")

                for scenario in scenarios:
                    self.run_scenario(scenario)

            # Finaliza episódio
            episode_result = self.end_episode()
            episode_elapsed = time.time() - episode_start

            # Coleta métricas
            metrics = self.get_metrics()
            metrics['episode'] = episode
            metrics['elapsed_time'] = episode_elapsed
            self.metrics_history.append(metrics)

            print(f"\n  ✅ Episódio {episode} finalizado em {episode_elapsed:.2f}s")
            print(f"  📊 Episode count: {episode_result.get('episode_count')}")

        print(f"\n{'='*70}")
        print("EXPERIMENTO CONCLUÍDO!")
        print(f"{'='*70}")

        # Desativa treinamento
        self.disable_training()
        print("✅ Modo de treinamento desativado")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Gera relatório completo do experimento"""
        print("\n📊 Gerando relatório...")

        # Análise por categoria de segurança
        by_security_level = {}
        for result in self.results:
            level = result['security_level']
            if level not in by_security_level:
                by_security_level[level] = []
            by_security_level[level].append(result)

        # Análise por presença de QKD
        with_qkd = [r for r in self.results if r['has_qkd']]
        without_qkd = [r for r in self.results if not r['has_qkd']]

        # Análise de algoritmos mais usados
        algorithm_usage = {}
        for result in self.results:
            for algo in result['proposed_algorithms']:
                algorithm_usage[algo] = algorithm_usage.get(algo, 0) + 1

        # Taxa de sucesso
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['feedback_success'])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        # Latência média
        latencies = [r['feedback_latency'] for r in self.results]
        avg_latency = statistics.mean(latencies) if latencies else 0
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        # Response time
        response_times = [r['response_time'] for r in self.results]
        avg_response_time = statistics.mean(response_times) if response_times else 0

        report = {
            'experiment_info': {
                'total_requests': total_requests,
                'total_episodes': len(self.metrics_history),
                'timestamp': datetime.now().isoformat()
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
            'qkd_analysis': {
                'with_qkd': {
                    'count': len(with_qkd),
                    'success_rate': sum(1 for r in with_qkd if r['feedback_success']) / len(with_qkd) * 100 if with_qkd else 0,
                    'avg_latency': statistics.mean([r['feedback_latency'] for r in with_qkd]) if with_qkd else 0
                },
                'without_qkd': {
                    'count': len(without_qkd),
                    'success_rate': sum(1 for r in without_qkd if r['feedback_success']) / len(without_qkd) * 100 if without_qkd else 0,
                    'avg_latency': statistics.mean([r['feedback_latency'] for r in without_qkd]) if without_qkd else 0
                }
            },
            'metrics_history': self.metrics_history,
            'raw_results': self.results
        }

        return report

    def save_results(self, report: Dict, prefix: str = "rl_experiment"):
        """Salva resultados em múltiplos formatos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON completo
        json_file = f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Relatório JSON salvo: {json_file}")

        # CSV com resultados detalhados
        csv_file = f"{prefix}_{timestamp}_details.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if report['raw_results']:
                writer = csv.DictWriter(f, fieldnames=report['raw_results'][0].keys())
                writer.writeheader()
                writer.writerows(report['raw_results'])
        print(f"✅ Detalhes CSV salvos: {csv_file}")

        # CSV com métricas por episódio
        metrics_csv = f"{prefix}_{timestamp}_metrics.csv"
        with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
            if report['metrics_history']:
                fieldnames = ['episode', 'elapsed_time']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(report['metrics_history'])
        print(f"✅ Métricas CSV salvas: {metrics_csv}")

        # Relatório resumido em texto
        txt_file = f"{prefix}_{timestamp}_summary.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RL ENGINE - RELATÓRIO EXPERIMENTAL\n")
            f.write("="*70 + "\n\n")

            f.write("INFORMAÇÕES DO EXPERIMENTO\n")
            f.write("-"*70 + "\n")
            for key, value in report['experiment_info'].items():
                f.write(f"{key}: {value}\n")

            f.write("\nMÉTRICAS DE PERFORMANCE\n")
            f.write("-"*70 + "\n")
            for key, value in report['performance_metrics'].items():
                f.write(f"{key}: {value:.2f}\n")

            f.write("\nUSO DE ALGORITMOS\n")
            f.write("-"*70 + "\n")
            sorted_algos = sorted(report['algorithm_usage'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for algo, count in sorted_algos:
                f.write(f"{algo}: {count} vezes\n")

            f.write("\nANÁLISE POR NÍVEL DE SEGURANÇA\n")
            f.write("-"*70 + "\n")
            for level, data in report['by_security_level'].items():
                f.write(f"\n{level.upper()}:\n")
                f.write(f"  Requisições: {data['count']}\n")
                f.write(f"  Taxa de sucesso: {data['success_rate']:.2f}%\n")
                f.write(f"  Latência média: {data['avg_latency']:.2f}ms\n")

            f.write("\nANÁLISE QKD\n")
            f.write("-"*70 + "\n")
            f.write("Com QKD:\n")
            f.write(f"  Requisições: {report['qkd_analysis']['with_qkd']['count']}\n")
            f.write(f"  Taxa de sucesso: {report['qkd_analysis']['with_qkd']['success_rate']:.2f}%\n")
            f.write(f"  Latência média: {report['qkd_analysis']['with_qkd']['avg_latency']:.2f}ms\n")
            f.write("\nSem QKD:\n")
            f.write(f"  Requisições: {report['qkd_analysis']['without_qkd']['count']}\n")
            f.write(f"  Taxa de sucesso: {report['qkd_analysis']['without_qkd']['success_rate']:.2f}%\n")
            f.write(f"  Latência média: {report['qkd_analysis']['without_qkd']['avg_latency']:.2f}ms\n")

        print(f"✅ Resumo TXT salvo: {txt_file}")

        return {
            'json': json_file,
            'csv': csv_file,
            'metrics_csv': metrics_csv,
            'summary': txt_file
        }


# Definição dos cenários de teste
SCENARIOS = [
    {
        'name': 'Ultra Security with QKD',
        'category': 'quantum',
        'context': {
            'request_id': 'exp-ultra-qkd-001',
            'source': 'node-A',
            'destination': 'http://localhost:9000',
            'security_level': 'ultra',
            'risk_score': 0.95,
            'conf_score': 0.98,
            'data_sensitivity': 0.95,
            'service_criticality': 0.98,
            'available_resources': 0.9,
            'dst_props': {
                'hardware': ['QKD', 'QUANTUM'],
                'compliance': ['GDPR', 'HIPAA']
            }
        }
    },
    {
        'name': 'High Security with QKD',
        'category': 'quantum',
        'context': {
            'request_id': 'exp-high-qkd-002',
            'source': 'node-B',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.85,
            'conf_score': 0.90,
            'data_sensitivity': 0.85,
            'available_resources': 0.8,
            'dst_props': {
                'hardware': ['QKD']
            }
        }
    },
    {
        'name': 'Very High Security with PQC',
        'category': 'post_quantum',
        'context': {
            'request_id': 'exp-veryhigh-pqc-003',
            'source': 'node-C',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.82,
            'conf_score': 0.88,
            'data_sensitivity': 0.90,
            'available_resources': 0.75,
            'dst_props': {
                'hardware': [],
                'compliance': ['GDPR', 'SOC2']
            }
        }
    },
    {
        'name': 'High Security Hybrid',
        'category': 'hybrid',
        'context': {
            'request_id': 'exp-high-hybrid-004',
            'source': 'node-D',
            'destination': 'http://localhost:9000',
            'security_level': 'very_high',
            'risk_score': 0.78,
            'conf_score': 0.82,
            'data_sensitivity': 0.88,
            'available_resources': 0.80,
            'dst_props': {
                'hardware': [],
                'compliance': ['FIPS-140-2']
            }
        }
    },
    {
        'name': 'Moderate Security Classical',
        'category': 'classical',
        'context': {
            'request_id': 'exp-moderate-classical-005',
            'source': 'node-E',
            'destination': 'http://localhost:9000',
            'security_level': 'moderate',
            'risk_score': 0.45,
            'conf_score': 0.50,
            'available_resources': 0.6,
            'dst_props': {
                'hardware': [],
                'compliance': ['PCI-DSS']
            }
        }
    },
    {
        'name': 'Low Security Minimal',
        'category': 'classical',
        'context': {
            'request_id': 'exp-low-minimal-006',
            'source': 'node-F',
            'destination': 'http://localhost:9000',
            'security_level': 'low',
            'risk_score': 0.15,
            'conf_score': 0.20,
            'available_resources': 0.5,
            'dst_props': {}
        }
    },
    {
        'name': 'Peak Attack Time with QKD',
        'category': 'stress_test',
        'context': {
            'request_id': 'exp-peak-qkd-007',
            'source': 'node-G',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.88,
            'conf_score': 0.85,
            'is_peak_attack_time': True,
            'current_threat_level': 0.92,
            'system_load': 0.85,
            'available_resources': 0.35,
            'dst_props': {
                'hardware': ['QKD']
            }
        }
    },
    {
        'name': 'Limited Resources High Security',
        'category': 'stress_test',
        'context': {
            'request_id': 'exp-limited-high-008',
            'source': 'node-H',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.75,
            'conf_score': 0.80,
            'system_load': 0.90,
            'available_resources': 0.25,
            'dst_props': {
                'hardware': []
            }
        }
    },
    {
        'name': 'High Latency Network',
        'category': 'network_conditions',
        'context': {
            'request_id': 'exp-highlatency-009',
            'source': 'node-I',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.65,
            'conf_score': 0.70,
            'network_latency': 250.0,
            'available_resources': 0.60,
            'dst_props': {
                'hardware': []
            }
        }
    },
    {
        'name': 'Night Weekend High Risk',
        'category': 'temporal',
        'context': {
            'request_id': 'exp-night-weekend-010',
            'source': 'node-J',
            'destination': 'http://localhost:9000',
            'security_level': 'high',
            'risk_score': 0.70,
            'conf_score': 0.75,
            'time_of_day': 2,
            'day_of_week': 6,
            'is_peak_attack_time': True,
            'dst_props': {
                'hardware': ['QKD']
            }
        }
    }
]


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("RL ENGINE - FRAMEWORK DE TESTES EXPERIMENTAIS")
    print("="*70)

    # Configuração do experimento
    experiment = RLEngineExperiment(base_url="http://localhost:9009")

    # Executa experimento
    report = experiment.run_experiment(
        scenarios=SCENARIOS,
        episodes=5,  # Número de episódios de treinamento
        iterations_per_episode=10  # Iterações por episódio
    )

    if report:
        # Salva resultados
        files = experiment.save_results(report, prefix="rl_experiment")

        print("\n" + "="*70)
        print("ARQUIVOS GERADOS:")
        print("="*70)
        for file_type, filename in files.items():
            print(f"  {file_type}: {filename}")

        print("\n" + "="*70)
        print("RESUMO RÁPIDO:")
        print("="*70)
        print(f"Taxa de sucesso: {report['performance_metrics']['success_rate']:.2f}%")
        print(f"Latência média: {report['performance_metrics']['avg_latency_ms']:.2f}ms")
        print(f"Tempo de resposta: {report['performance_metrics']['avg_response_time_s']:.4f}s")
        print(f"Total de requisições: {report['experiment_info']['total_requests']}")

        print("\n✅ Experimento concluído com sucesso!")
        print("\n📊 Use os arquivos gerados para análise e construção do artigo.")


if __name__ == "__main__":
    main()
