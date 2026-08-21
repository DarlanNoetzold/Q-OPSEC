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



               

API_BASE_URL = "http://localhost:8083/api/v2/trust"

SCENARIOS_FILE = "trust_test_scenarios.json"

RESULTS_FILE = "trust_test_results.json"

USE_ARTIFICIAL_DATA = True                                             





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



    def generate_artificial_result(self, scenario: Dict[str, Any], index: int) -> Dict[str, Any]:

        """Gera resultado artificial com dados realistas"""

                                               

        trust_scores = [0.85, 0.62, 0.45, 0.78, 0.71, 0.38, 0.55, 0.82, 0.76, 0.58]

        trust_levels = ['HIGH', 'MEDIUM', 'MEDIUM', 'HIGH', 'HIGH', 'LOW', 'MEDIUM', 'HIGH', 'HIGH', 'MEDIUM']



        score = trust_scores[index % len(trust_scores)]

        level = trust_levels[index % len(trust_levels)]



                                  

        dimensions = {

            'temporal': round(np.random.uniform(0.05, 0.15), 4),

            'source': round(np.random.uniform(0.04, 0.12), 4),

            'semantic': round(np.random.uniform(0.05, 0.13), 4),

            'anomaly': round(np.random.uniform(0.02, 0.08), 4),

            'consistency': round(np.random.uniform(0.04, 0.10), 4),

            'context': round(np.random.uniform(0.05, 0.12), 4)

        }



                                                 

        signals = [

            {

                'name': 'temporal',

                'score': round(np.random.uniform(0.7, 1.0), 2),

                'confidence': round(np.random.uniform(0.8, 1.0), 1),

                'metadata': {}

            },

            {

                'name': 'temporal_drift',

                'score': round(np.random.uniform(0.5, 0.9), 2),

                'confidence': round(np.random.uniform(0.6, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'source_reliability',

                'score': round(np.random.uniform(0.6, 0.9), 2),

                'confidence': round(np.random.uniform(0.7, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'source_consistency',

                'score': round(np.random.uniform(0.6, 0.9), 2),

                'confidence': round(np.random.uniform(0.6, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'semantic_consistency',

                'score': round(np.random.uniform(0.6, 0.9), 2),

                'confidence': round(np.random.uniform(0.6, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'semantic_drift',

                'score': round(np.random.uniform(0.7, 0.9), 2),

                'confidence': round(np.random.uniform(0.6, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'anomaly_detection',

                'score': round(np.random.uniform(0.5, 0.9), 2),

                'confidence': round(np.random.uniform(0.5, 0.8), 1),

                'metadata': {}

            },

            {

                'name': 'consistency',

                'score': round(np.random.uniform(0.6, 0.9), 2),

                'confidence': round(np.random.uniform(0.6, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'context_alignment',

                'score': round(np.random.uniform(0.5, 0.9), 2),

                'confidence': round(np.random.uniform(0.7, 0.9), 1),

                'metadata': {}

            },

            {

                'name': 'context_stability',

                'score': round(np.random.uniform(0.6, 0.9), 2),

                'confidence': round(np.random.uniform(0.6, 0.9), 1),

                'metadata': {}

            }

        ]



                                     

        details = []

        for signal in signals:

            contribution = round(np.random.uniform(0.02, 0.18), 3)

            details.append({

                'signal': signal['name'],

                'score': signal['score'],

                'confidence': signal['confidence'],

                'contribution': contribution,

                'weight': round(np.random.uniform(0.05, 0.18), 3),

                'explanation': f"Signal {signal['name']} evaluated successfully",

                'metadata': signal['metadata']

            })



        result = {

            'trust_score': score,

            'trust_level': level,

            'confidence_interval': {

                'lower': round(max(0, score - 0.08), 4),

                'upper': round(min(1.0, score + 0.08), 4)

            },

            'dimensions': dimensions,

            'signals': signals,

            'explainability': {

                'summary': f"Trust evaluation completed with {level} trust level",

                'details': details,

                'top_factors': [

                    {'signal': details[0]['signal'], 'contribution': details[0]['contribution'], 'impact': 'positive'},

                    {'signal': details[1]['signal'], 'contribution': details[1]['contribution'], 'impact': 'positive'},

                    {'signal': details[2]['signal'], 'contribution': details[2]['contribution'], 'impact': 'neutral'}

                ],

                'recommendations': ['Standard validation procedures apply']

            },

            'risk_flags': [],

            'trust_dna': f"T-DNA-v2:{np.random.randint(10000000, 99999999):08x}",

            'metadata': {

                'source_id': scenario['metadata']['source_id'],

                'entity_id': scenario['metadata']['entity_id'],

                'timestamp': datetime.now().isoformat(),

                'data_type': scenario['metadata'].get('data_type', 'unknown'),

                'environment': scenario['metadata'].get('environment', 'unknown'),

                'num_signals': len(signals)

            }

        }



        return result



    def run_scenario(self, scenario: Dict[str, Any], index: int) -> Dict[str, Any]:

        """Executa um cenário de teste"""

        print(f"\n🧪 Executando: {scenario['name']}")



        if USE_ARTIFICIAL_DATA:

                                   

            time.sleep(0.05)                   

            elapsed_time = np.random.uniform(0.03, 0.08)

            result = self.generate_artificial_result(scenario, index)

            print(f"   ✅ Trust Score: {result['trust_score']:.4f} | Level: {result['trust_level']} (artificial)")



            return {

                "scenario_name": scenario['name'],

                "description": scenario['description'],

                "success": True,

                "status_code": 200,

                "elapsed_time": elapsed_time,

                "result": result

            }



                                                   

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



        for index, scenario in enumerate(scenarios):

            result = self.run_scenario(scenario, index)

            self.results.append(result)

            time.sleep(0.1)                                



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

            "artificial_data": USE_ARTIFICIAL_DATA,

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

        scenarios = [r['scenario_name'].replace('Scenario ', 'S').replace(': ', '\n') for r in self.results]

        scores = [r['result']['trust_score'] for r in self.results]

        levels = [r['result']['trust_level'] for r in self.results]



                               

        color_map = {

            'VERY_LOW': '#d32f2f',

            'LOW': '#f57c00',

            'MEDIUM': '#fbc02d',

            'HIGH': '#689f38',

            'VERY_HIGH': '#388e3c'

        }

        colors = [color_map.get(level, '#757575') for level in levels]



        plt.figure(figsize=(16, 7))

        bars = plt.bar(range(len(scenarios)), scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)



                                     

        plt.axhline(y=0.8, color='#388e3c', linestyle='--', linewidth=2, alpha=0.7, label='High (0.8)')

        plt.axhline(y=0.6, color='#f57c00', linestyle='--', linewidth=2, alpha=0.7, label='Medium (0.6)')

        plt.axhline(y=0.4, color='#d32f2f', linestyle='--', linewidth=2, alpha=0.7, label='Low (0.4)')



        plt.xlabel('Scenarios', fontsize=13, fontweight='bold')

        plt.ylabel('Trust Score', fontsize=13, fontweight='bold')

        plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right', fontsize=9)

        plt.ylim(0, 1.0)

        plt.grid(axis='y', alpha=0.3, linestyle='--')

        plt.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        plt.savefig('trust_scores_by_scenario.png', dpi=300, bbox_inches='tight', facecolor='white')

        print("📊 Gráfico salvo: trust_scores_by_scenario.png")

        plt.close()



    def plot_signal_contributions(self):

        """Gráfico de contribuições médias dos signals"""

        signal_contributions = defaultdict(list)



        for result in self.results:

            for detail in result['result']['explainability']['details']:

                signal_contributions[detail['signal']].append(detail['contribution'])



                        

        signals = list(signal_contributions.keys())

        avg_contributions = [np.mean(signal_contributions[s]) for s in signals]



                                 

        sorted_data = sorted(zip(signals, avg_contributions), key=lambda x: x[1], reverse=True)

        signals, avg_contributions = zip(*sorted_data)



                          

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(signals)))



        plt.figure(figsize=(12, 7))

        bars = plt.barh(signals, avg_contributions, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)



        plt.xlabel('Average Contribution', fontsize=13, fontweight='bold')

        plt.ylabel('Signals', fontsize=13, fontweight='bold')

        plt.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        plt.savefig('signal_contributions.png', dpi=300, bbox_inches='tight', facecolor='white')

        print("📊 Gráfico salvo: signal_contributions.png")

        plt.close()



    def plot_trust_dimensions(self):

        """Gráfico radar das dimensões de trust"""

        if not self.results:

            return



                                                            

        all_dimensions = defaultdict(list)

        for result in self.results:

            for dim, value in result['result']['dimensions'].items():

                all_dimensions[dim].append(value)



        categories = list(all_dimensions.keys())

        values = [np.mean(all_dimensions[cat]) for cat in categories]



                          

        values += values[:1]

        categories += categories[:1]



        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)



        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=3, color='#1f77b4', markersize=8)

        ax.fill(angles, values, alpha=0.3, color='#1f77b4')

        ax.set_xticks(angles[:-1])

        ax.set_xticklabels(categories[:-1], fontsize=11, fontweight='bold')

        ax.set_ylim(0, max(values) * 1.3)

        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        plt.savefig('trust_dimensions_radar.png', dpi=300, bbox_inches='tight', facecolor='white')

        print("📊 Gráfico salvo: trust_dimensions_radar.png")

        plt.close()



    def plot_performance_metrics(self):

        """Gráfico de métricas de performance"""

        scenarios = [r['scenario_name'].replace('Scenario ', 'S').replace(': ', '\n') for r in self.results]

        elapsed_times = [r['elapsed_time'] * 1000 for r in self.results]      



        plt.figure(figsize=(16, 7))

        bars = plt.bar(range(len(scenarios)), elapsed_times, color='#ff7043', alpha=0.85, edgecolor='black', linewidth=1.5)



                        

        avg_time = np.mean(elapsed_times)

        plt.axhline(y=avg_time, color='#d32f2f', linestyle='--', linewidth=2.5, 

                   label=f'Average: {avg_time:.2f}ms', alpha=0.8)



        plt.xlabel('Scenarios', fontsize=13, fontweight='bold')

        plt.ylabel('Response Time (ms)', fontsize=13, fontweight='bold')

        plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right', fontsize=9)

        plt.grid(axis='y', alpha=0.3, linestyle='--')

        plt.legend(loc='upper right', fontsize=11)

        plt.tight_layout()

        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')

        print("📊 Gráfico salvo: performance_metrics.png")

        plt.close()



    def plot_confidence_intervals(self):

        """Gráfico de intervalos de confiança"""

        scenarios = [r['scenario_name'].replace('Scenario ', 'S').replace(': ', '\n') for r in self.results]

        scores = [r['result']['trust_score'] for r in self.results]

        lower_bounds = [r['result']['confidence_interval']['lower'] for r in self.results]

        upper_bounds = [r['result']['confidence_interval']['upper'] for r in self.results]



        x = range(len(scenarios))



        plt.figure(figsize=(16, 7))

        plt.errorbar(x, scores, 

                     yerr=[np.array(scores) - np.array(lower_bounds), 

                           np.array(upper_bounds) - np.array(scores)],

                     fmt='o', markersize=10, capsize=6, capthick=2.5, 

                     color='#1f77b4', ecolor='#7f7f7f', elinewidth=2, alpha=0.85)



        plt.xlabel('Scenarios', fontsize=13, fontweight='bold')

        plt.ylabel('Trust Score', fontsize=13, fontweight='bold')

        plt.xticks(x, scenarios, rotation=45, ha='right', fontsize=9)

        plt.ylim(0, 1.0)

        plt.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight', facecolor='white')

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



    if USE_ARTIFICIAL_DATA:

        print("⚠️  MODO: Dados Artificiais (para demonstração)")

    else:

        print("🔗 MODO: API Real")



    print("="*70)



                       

    runner = TrustTestRunner(API_BASE_URL)

    scenarios = runner.load_scenarios(SCENARIOS_FILE)

    runner.run_all_scenarios(scenarios)

    runner.save_results(RESULTS_FILE)



                                 

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
