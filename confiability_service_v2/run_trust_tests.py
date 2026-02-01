"""
Trust Engine V2 - Test Runner & Analytics
Executa cenários de teste e gera gráficos/métricas
"""
import json
import requests
import time
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configurações
API_BASE_URL = "http://localhost:8083/api/v2/trust"
SCENARIOS_FILE = "trust_test_scenarios.json"
RESULTS_FILE = "trust_test_results.json"


class TrustTestRunner:
    """
    Executa testes e coleta resultados
    """

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.results = []

    def load_scenarios(self, file_path: str) -> List[Dict[str, Any]]:
        """Carrega cenários de teste do JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['scenarios']

    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um cenário de teste"""
        print(f"\n🧪 Executando: {scenario['name']}")

        request_data = {
            "payload": scenario['payload'],
            "metadata": scenario['metadata']
        }

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_url}/evaluate",
                json=request_data,
                timeout=10
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Trust Score: {result['trust_score']:.4f} | Level: {result['trust_level']}")

                return {
                    "scenario_name": scenario['name'],
                    "description": scenario['description'],
                    "success": True,
                    "status_code": 200,
                    "elapsed_time": elapsed_time,
                    "result": result
                }
            else:
                print(f"   ❌ Erro: {response.status_code}")
                return {
                    "scenario_name": scenario['name'],
                    "description": scenario['description'],
                    "success": False,
                    "status_code": response.status_code,
                    "elapsed_time": elapsed_time,
                    "error": response.text
                }

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ Exception: {str(e)}")
            return {
                "scenario_name": scenario['name'],
                "description": scenario['description'],
                "success": False,
                "status_code": 0,
                "elapsed_time": elapsed_time,
                "error": str(e)
            }

    def run_all_scenarios(self, scenarios: List[Dict[str, Any]]):
        """Executa todos os cenários"""
        print("\n" + "="*70)
        print("🚀 INICIANDO TESTES DO TRUST ENGINE V2")
        print("="*70)

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            self.results.append(result)
            time.sleep(0.5)  # Pequeno delay entre requests

        print("\n" + "="*70)
        print("✅ TESTES CONCLUÍDOS")
        print("="*70)

    def save_results(self, file_path: str):
        """Salva resultados em JSON"""
        output = {
            "test_run_timestamp": datetime.now().isoformat(),
            "total_scenarios": len(self.results),
            "successful": sum(1 for r in self.results if r['success']),
            "failed": sum(1 for r in self.results if not r['success']),
            "results": self.results
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resultados salvos em: {file_path}")


class TrustAnalytics:
    """
    Gera gráficos e métricas dos resultados
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = [r for r in results if r['success']]

    def plot_trust_scores(self):
        """Gráfico de Trust Scores por cenário"""
        scenarios = [r['scenario_name'].replace('Scenario ', 'S') for r in self.results]
        scores = [r['result']['trust_score'] for r in self.results]
        levels = [r['result']['trust_level'] for r in self.results]

        # Cores por trust level
        color_map = {
            'VERY_LOW': '#d32f2f',
            'LOW': '#f57c00',
            'MEDIUM': '#fbc02d',
            'HIGH': '#689f38',
            'VERY_HIGH': '#388e3c'
        }
        colors = [color_map.get(level, '#757575') for level in levels]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(scenarios)), scores, color=colors, alpha=0.8, edgecolor='black')

        # Adiciona linha de threshold
        plt.axhline(y=0.8, color='green', linestyle='--', linewidth=1, label='High Threshold (0.8)')
        plt.axhline(y=0.6, color='orange', linestyle='--', linewidth=1, label='Medium Threshold (0.6)')
        plt.axhline(y=0.4, color='red', linestyle='--', linewidth=1, label='Low Threshold (0.4)')

        plt.xlabel('Cenários', fontsize=12, fontweight='bold')
        plt.ylabel('Trust Score', fontsize=12, fontweight='bold')
        plt.title('Trust Scores por Cenário de Teste', fontsize=14, fontweight='bold')
        plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('trust_scores_by_scenario.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico salvo: trust_scores_by_scenario.png")
        plt.close()

    def plot_signal_contributions(self):
        """Gráfico de contribuições médias dos signals"""
        signal_contributions = defaultdict(list)

        for result in self.results:
            for detail in result['result']['explainability']['details']:
                signal_contributions[detail['signal']].append(detail['contribution'])

        # Calcula médias
        signals = list(signal_contributions.keys())
        avg_contributions = [np.mean(signal_contributions[s]) for s in signals]

        # Ordena por contribuição
        sorted_data = sorted(zip(signals, avg_contributions), key=lambda x: x[1], reverse=True)
        signals, avg_contributions = zip(*sorted_data)

        plt.figure(figsize=(12, 6))
        bars = plt.barh(signals, avg_contributions, color='steelblue', alpha=0.8, edgecolor='black')

        plt.xlabel('Contribuição Média', fontsize=12, fontweight='bold')
        plt.ylabel('Signals', fontsize=12, fontweight='bold')
        plt.title('Contribuição Média dos Signals', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('signal_contributions.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico salvo: signal_contributions.png")
        plt.close()

    def plot_trust_dimensions(self):
        """Gráfico radar das dimensões de trust"""
        # Pega o primeiro resultado como exemplo
        if not self.results:
            return

        dimensions = self.results[0]['result']['dimensions']
        categories = list(dimensions.keys())
        values = [dimensions[cat] for cat in categories]

        # Fecha o polígono
        values += values[:1]
        categories += categories[:1]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue', label='Dimensões')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title('Dimensões de Trust (Exemplo)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('trust_dimensions_radar.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico salvo: trust_dimensions_radar.png")
        plt.close()

    def plot_performance_metrics(self):
        """Gráfico de métricas de performance"""
        scenarios = [r['scenario_name'].replace('Scenario ', 'S') for r in self.results]
        elapsed_times = [r['elapsed_time'] * 1000 for r in self.results]  # ms

        plt.figure(figsize=(14, 6))
        plt.bar(range(len(scenarios)), elapsed_times, color='coral', alpha=0.8, edgecolor='black')

        # Linha de média
        avg_time = np.mean(elapsed_times)
        plt.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=f'Média: {avg_time:.2f}ms')

        plt.xlabel('Cenários', fontsize=12, fontweight='bold')
        plt.ylabel('Tempo de Resposta (ms)', fontsize=12, fontweight='bold')
        plt.title('Performance: Tempo de Resposta por Cenário', fontsize=14, fontweight='bold')
        plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico salvo: performance_metrics.png")
        plt.close()

    def plot_confidence_intervals(self):
        """Gráfico de intervalos de confiança"""
        scenarios = [r['scenario_name'].replace('Scenario ', 'S') for r in self.results]
        scores = [r['result']['trust_score'] for r in self.results]
        lower_bounds = [r['result']['confidence_interval']['lower'] for r in self.results]
        upper_bounds = [r['result']['confidence_interval']['upper'] for r in self.results]

        x = range(len(scenarios))

        plt.figure(figsize=(14, 6))
        plt.errorbar(x, scores, 
                     yerr=[np.array(scores) - np.array(lower_bounds), 
                           np.array(upper_bounds) - np.array(scores)],
                     fmt='o', markersize=8, capsize=5, capthick=2, 
                     color='steelblue', ecolor='gray', alpha=0.8)

        plt.xlabel('Cenários', fontsize=12, fontweight='bold')
        plt.ylabel('Trust Score', fontsize=12, fontweight='bold')
        plt.title('Trust Scores com Intervalos de Confiança', fontsize=14, fontweight='bold')
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico salvo: confidence_intervals.png")
        plt.close()

    def generate_summary_report(self):
        """Gera relatório resumido em texto"""
        print("\n" + "="*70)
        print("📊 RELATÓRIO DE ANÁLISE")
        print("="*70)

        scores = [r['result']['trust_score'] for r in self.results]
        times = [r['elapsed_time'] * 1000 for r in self.results]

        print(f"\n📈 TRUST SCORES:")
        print(f"   Média:    {np.mean(scores):.4f}")
        print(f"   Mediana:  {np.median(scores):.4f}")
        print(f"   Mínimo:   {np.min(scores):.4f}")
        print(f"   Máximo:   {np.max(scores):.4f}")
        print(f"   Desvio:   {np.std(scores):.4f}")

        print(f"\n⚡ PERFORMANCE:")
        print(f"   Tempo médio:   {np.mean(times):.2f}ms")
        print(f"   Tempo mínimo:  {np.min(times):.2f}ms")
        print(f"   Tempo máximo:  {np.max(times):.2f}ms")

        # Distribuição por trust level
        levels = [r['result']['trust_level'] for r in self.results]
        level_counts = defaultdict(int)
        for level in levels:
            level_counts[level] += 1

        print(f"\n🎯 DISTRIBUIÇÃO POR TRUST LEVEL:")
        for level, count in sorted(level_counts.items()):
            percentage = (count / len(levels)) * 100
            print(f"   {level:12s}: {count:2d} ({percentage:5.1f}%)")

        print("\n" + "="*70)


def main():
    """Função principal"""
    print("\n🔍 TRUST ENGINE V2 - TEST RUNNER & ANALYTICS")
    print("="*70)

    # 1. Executa testes
    runner = TrustTestRunner(API_BASE_URL)
    scenarios = runner.load_scenarios(SCENARIOS_FILE)
    runner.run_all_scenarios(scenarios)
    runner.save_results(RESULTS_FILE)

    # 2. Gera análises e gráficos
    print("\n📊 Gerando gráficos e métricas...")
    analytics = TrustAnalytics(runner.results)

    analytics.plot_trust_scores()
    analytics.plot_signal_contributions()
    analytics.plot_trust_dimensions()
    analytics.plot_performance_metrics()
    analytics.plot_confidence_intervals()
    analytics.generate_summary_report()

    print("\n✅ Análise completa!")
    print("\n📁 Arquivos gerados:")
    print("   - trust_test_results.json")
    print("   - trust_scores_by_scenario.png")
    print("   - signal_contributions.png")
    print("   - trust_dimensions_radar.png")
    print("   - performance_metrics.png")
    print("   - confidence_intervals.png")


if __name__ == "__main__":
    main()
