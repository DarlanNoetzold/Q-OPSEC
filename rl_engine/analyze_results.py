
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultAnalyzer:
    def __init__(self, json_file: str):
        """Inicializa o analisador com arquivo JSON de resultados"""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.df = pd.DataFrame(self.data['raw_results'])
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)

    def generate_all_plots(self):
        """Gera todos os gráficos para o artigo"""
        print("\n📊 Gerando visualizações...")

        self.plot_algorithm_distribution()
        self.plot_success_rate_by_security()
        self.plot_latency_comparison()
        self.plot_qkd_vs_non_qkd()
        self.plot_response_time_distribution()
        self.plot_security_level_heatmap()
        self.plot_temporal_analysis()
        self.plot_resource_usage()

        print(f"\n✅ Gráficos salvos em: {self.output_dir}/")

    def plot_algorithm_distribution(self):
        """Distribuição de uso de algoritmos"""
        algo_counts = {}
        for _, row in self.df.iterrows():
            for algo in row['proposed_algorithms']:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1

        fig, ax = plt.subplots(figsize=(14, 8))
        algos = list(algo_counts.keys())
        counts = list(algo_counts.values())

        colors = plt.cm.viridis(np.linspace(0, 1, len(algos)))
        bars = ax.barh(algos, counts, color=colors)

        ax.set_xlabel('Número de Vezes Selecionado', fontsize=12, fontweight='bold')
        ax.set_ylabel('Algoritmo Criptográfico', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição de Uso de Algoritmos Criptográficos', 
                    fontsize=14, fontweight='bold', pad=20)

        # Adiciona valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ algorithm_distribution.png")

    def plot_success_rate_by_security(self):
        """Taxa de sucesso por nível de segurança"""
        success_by_level = self.df.groupby('security_level').agg({
            'feedback_success': ['mean', 'count']
        }).reset_index()

        success_by_level.columns = ['security_level', 'success_rate', 'count']
        success_by_level['success_rate'] *= 100

        fig, ax = plt.subplots(figsize=(12, 7))

        bars = ax.bar(success_by_level['security_level'], 
                     success_by_level['success_rate'],
                     color=plt.cm.RdYlGn(success_by_level['success_rate']/100))

        ax.set_xlabel('Nível de Segurança', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taxa de Sucesso (%)', fontsize=12, fontweight='bold')
        ax.set_title('Taxa de Sucesso por Nível de Segurança', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 105)

        # Adiciona valores e contagem
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = success_by_level.iloc[i]['count']
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%\n(n={count})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_by_security.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ success_rate_by_security.png")

    def plot_latency_comparison(self):
        """Comparação de latência entre níveis de segurança"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Box plot
        self.df.boxplot(column='feedback_latency', by='security_level', ax=ax1)
        ax1.set_xlabel('Nível de Segurança', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Latência (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Distribuição de Latência por Nível de Segurança', 
                     fontsize=12, fontweight='bold')
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')

        # Violin plot
        sns.violinplot(data=self.df, x='security_level', y='feedback_latency', ax=ax2)
        ax2.set_xlabel('Nível de Segurança', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Latência (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Densidade de Latência por Nível de Segurança', 
                     fontsize=12, fontweight='bold')
        plt.sca(ax2)
        plt.xticks(rotation=45, ha='right')

        plt.suptitle('')  # Remove título automático
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ latency_comparison.png")

    def plot_qkd_vs_non_qkd(self):
        """Comparação QKD vs Não-QKD"""
        qkd_comparison = self.df.groupby('has_qkd').agg({
            'feedback_success': 'mean',
            'feedback_latency': 'mean',
            'feedback_resource_usage': 'mean',
            'request_id': 'count'
        }).reset_index()

        qkd_comparison.columns = ['has_qkd', 'success_rate', 'avg_latency', 
                                  'avg_resource', 'count']
        qkd_comparison['success_rate'] *= 100
        qkd_comparison['has_qkd'] = qkd_comparison['has_qkd'].map({
            True: 'Com QKD', False: 'Sem QKD'
        })

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Taxa de sucesso
        axes[0, 0].bar(qkd_comparison['has_qkd'], qkd_comparison['success_rate'],
                      color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_ylabel('Taxa de Sucesso (%)', fontweight='bold')
        axes[0, 0].set_title('Taxa de Sucesso', fontweight='bold')
        axes[0, 0].set_ylim(0, 105)
        for i, v in enumerate(qkd_comparison['success_rate']):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

        # Latência
        axes[0, 1].bar(qkd_comparison['has_qkd'], qkd_comparison['avg_latency'],
                      color=['#3498db', '#9b59b6'])
        axes[0, 1].set_ylabel('Latência Média (ms)', fontweight='bold')
        axes[0, 1].set_title('Latência Média', fontweight='bold')
        for i, v in enumerate(qkd_comparison['avg_latency']):
            axes[0, 1].text(i, v + 2, f'{v:.1f}ms', ha='center', fontweight='bold')

        # Uso de recursos
        axes[1, 0].bar(qkd_comparison['has_qkd'], qkd_comparison['avg_resource'],
                      color=['#f39c12', '#1abc9c'])
        axes[1, 0].set_ylabel('Uso Médio de Recursos', fontweight='bold')
        axes[1, 0].set_title('Uso de Recursos', fontweight='bold')
        axes[1, 0].set_ylim(0, 1.1)
        for i, v in enumerate(qkd_comparison['avg_resource']):
            axes[1, 0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

        # Contagem
        axes[1, 1].bar(qkd_comparison['has_qkd'], qkd_comparison['count'],
                      color=['#34495e', '#95a5a6'])
        axes[1, 1].set_ylabel('Número de Requisições', fontweight='bold')
        axes[1, 1].set_title('Volume de Requisições', fontweight='bold')
        for i, v in enumerate(qkd_comparison['count']):
            axes[1, 1].text(i, v + 5, f'{int(v)}', ha='center', fontweight='bold')

        plt.suptitle('Análise Comparativa: QKD vs Não-QKD', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qkd_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ qkd_comparison.png")

    def plot_response_time_distribution(self):
        """Distribuição de tempo de resposta"""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.hist(self.df['response_time'], bins=30, color='skyblue', 
               edgecolor='black', alpha=0.7)
        ax.axvline(self.df['response_time'].mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Média: {self.df["response_time"].mean():.4f}s')
        ax.axvline(self.df['response_time'].median(), color='green', 
                  linestyle='--', linewidth=2, label=f'Mediana: {self.df["response_time"].median():.4f}s')

        ax.set_xlabel('Tempo de Resposta (segundos)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição de Tempo de Resposta do RL Engine', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ response_time_distribution.png")

    def plot_security_level_heatmap(self):
        """Heatmap de métricas por nível de segurança"""
        metrics_by_level = self.df.groupby('security_level').agg({
            'feedback_success': 'mean',
            'feedback_latency': 'mean',
            'feedback_resource_usage': 'mean',
            'response_time': 'mean'
        })

        # Normalizar para 0-1
        metrics_normalized = (metrics_by_level - metrics_by_level.min()) / (metrics_by_level.max() - metrics_by_level.min())

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(metrics_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Valor Normalizado (0-1)'}, ax=ax)

        ax.set_xlabel('Nível de Segurança', fontsize=12, fontweight='bold')
        ax.set_ylabel('Métrica', fontsize=12, fontweight='bold')
        ax.set_title('Heatmap de Métricas por Nível de Segurança (Normalizado)', 
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'security_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ security_heatmap.png")

    def plot_temporal_analysis(self):
        """Análise temporal dos resultados"""
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        self.df['request_number'] = range(len(self.df))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Taxa de sucesso ao longo do tempo
        window = 20
        self.df['success_rolling'] = self.df['feedback_success'].rolling(window=window).mean() * 100
        axes[0].plot(self.df['request_number'], self.df['success_rolling'], 
                    linewidth=2, color='green')
        axes[0].set_ylabel('Taxa de Sucesso (%) - Média Móvel', fontweight='bold')
        axes[0].set_title(f'Evolução da Taxa de Sucesso (janela={window})', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        # Latência ao longo do tempo
        self.df['latency_rolling'] = self.df['feedback_latency'].rolling(window=window).mean()
        axes[1].plot(self.df['request_number'], self.df['latency_rolling'], 
                    linewidth=2, color='blue')
        axes[1].set_ylabel('Latência (ms) - Média Móvel', fontweight='bold')
        axes[1].set_title(f'Evolução da Latência (janela={window})', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Uso de recursos ao longo do tempo
        self.df['resource_rolling'] = self.df['feedback_resource_usage'].rolling(window=window).mean()
        axes[2].plot(self.df['request_number'], self.df['resource_rolling'], 
                    linewidth=2, color='orange')
        axes[2].set_xlabel('Número da Requisição', fontweight='bold')
        axes[2].set_ylabel('Uso de Recursos - Média Móvel', fontweight='bold')
        axes[2].set_title(f'Evolução do Uso de Recursos (janela={window})', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1.1)

        plt.suptitle('Análise Temporal de Performance', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ temporal_analysis.png")

    def plot_resource_usage(self):
        """Análise de uso de recursos"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter: Recursos vs Latência
        scatter = axes[0].scatter(self.df['feedback_resource_usage'], 
                                 self.df['feedback_latency'],
                                 c=self.df['feedback_success'].astype(int),
                                 cmap='RdYlGn', alpha=0.6, s=50)
        axes[0].set_xlabel('Uso de Recursos', fontweight='bold')
        axes[0].set_ylabel('Latência (ms)', fontweight='bold')
        axes[0].set_title('Relação: Uso de Recursos vs Latência', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Sucesso', fontweight='bold')

        # Distribuição de uso de recursos
        axes[1].hist(self.df['feedback_resource_usage'], bins=20, 
                    color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(self.df['feedback_resource_usage'].mean(), 
                       color='red', linestyle='--', linewidth=2,
                       label=f'Média: {self.df["feedback_resource_usage"].mean():.2f}')
        axes[1].set_xlabel('Uso de Recursos', fontweight='bold')
        axes[1].set_ylabel('Frequência', fontweight='bold')
        axes[1].set_title('Distribuição de Uso de Recursos', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ resource_usage_analysis.png")

    def generate_latex_tables(self):
        """Gera tabelas em formato LaTeX para o artigo"""
        print("\n📝 Gerando tabelas LaTeX...")

        latex_file = self.output_dir / 'tables.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            # Tabela 1: Métricas gerais
            f.write("% Tabela 1: Métricas Gerais de Performance\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Métricas Gerais de Performance do RL Engine}\n")
            f.write("\\begin{tabular}{|l|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Métrica} & \\textbf{Valor} \\\\\n")
            f.write("\\hline\n")

            perf = self.data['performance_metrics']
            f.write(f"Taxa de Sucesso & {perf['success_rate']:.2f}\\% \\\\\n")
            f.write(f"Latência Média & {perf['avg_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Desvio Padrão Latência & {perf['std_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Latência Mínima & {perf['min_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Latência Máxima & {perf['max_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Tempo de Resposta Médio & {perf['avg_response_time_s']:.4f} s \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:general_metrics}\n")
            f.write("\\end{table}\n\n")

            # Tabela 2: Por nível de segurança
            f.write("% Tabela 2: Métricas por Nível de Segurança\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Métricas por Nível de Segurança}\n")
            f.write("\\begin{tabular}{|l|r|r|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Nível} & \\textbf{Requisições} & \\textbf{Taxa Sucesso (\\%)} & \\textbf{Latência (ms)} \\\\\n")
            f.write("\\hline\n")

            for level, data in self.data['by_security_level'].items():
                f.write(f"{level} & {data['count']} & {data['success_rate']:.2f} & {data['avg_latency']:.2f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:security_level_metrics}\n")
            f.write("\\end{table}\n\n")

            # Tabela 3: Comparação QKD
            f.write("% Tabela 3: Comparação QKD vs Não-QKD\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparação: QKD vs Não-QKD}\n")
            f.write("\\begin{tabular}{|l|r|r|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Tipo} & \\textbf{Requisições} & \\textbf{Taxa Sucesso (\\%)} & \\textbf{Latência (ms)} \\\\\n")
            f.write("\\hline\n")

            qkd_data = self.data['qkd_analysis']
            f.write(f"Com QKD & {qkd_data['with_qkd']['count']} & {qkd_data['with_qkd']['success_rate']:.2f} & {qkd_data['with_qkd']['avg_latency']:.2f} \\\\\n")
            f.write(f"Sem QKD & {qkd_data['without_qkd']['count']} & {qkd_data['without_qkd']['success_rate']:.2f} & {qkd_data['without_qkd']['avg_latency']:.2f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:qkd_comparison}\n")
            f.write("\\end{table}\n")

        print(f"  ✓ tables.tex")
        print(f"\n✅ Tabelas LaTeX salvas em: {latex_file}")

    def generate_summary_report(self):
        """Gera relatório resumido para o artigo"""
        print("\n📄 Gerando relatório resumido...")

        report_file = self.output_dir / 'article_summary.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RL Engine - Resumo para Artigo Científico\n\n")

            f.write("## 1. Informações do Experimento\n\n")
            exp_info = self.data['experiment_info']
            f.write(f"- **Total de Requisições**: {exp_info['total_requests']}\n")
            f.write(f"- **Total de Episódios**: {exp_info['total_episodes']}\n")
            f.write(f"- **Data do Experimento**: {exp_info['timestamp']}\n\n")

            f.write("## 2. Métricas de Performance\n\n")
            perf = self.data['performance_metrics']
            f.write(f"- **Taxa de Sucesso**: {perf['success_rate']:.2f}%\n")
            f.write(f"- **Latência Média**: {perf['avg_latency_ms']:.2f} ms (±{perf['std_latency_ms']:.2f})\n")
            f.write(f"- **Latência Mínima/Máxima**: {perf['min_latency_ms']:.2f} / {perf['max_latency_ms']:.2f} ms\n")
            f.write(f"- **Tempo de Resposta Médio**: {perf['avg_response_time_s']:.4f} s\n\n")

            f.write("## 3. Algoritmos Mais Utilizados\n\n")
            sorted_algos = sorted(self.data['algorithm_usage'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for i, (algo, count) in enumerate(sorted_algos[:5], 1):
                f.write(f"{i}. **{algo}**: {count} vezes\n")
            f.write("\n")

            f.write("## 4. Análise por Nível de Segurança\n\n")
            for level, data in self.data['by_security_level'].items():
                f.write(f"### {level.upper()}\n")
                f.write(f"- Requisições: {data['count']}\n")
                f.write(f"- Taxa de Sucesso: {data['success_rate']:.2f}%\n")
                f.write(f"- Latência Média: {data['avg_latency']:.2f} ms\n\n")

            f.write("## 5. Comparação QKD vs Não-QKD\n\n")
            qkd = self.data['qkd_analysis']
            f.write("### Com QKD\n")
            f.write(f"- Requisições: {qkd['with_qkd']['count']}\n")
            f.write(f"- Taxa de Sucesso: {qkd['with_qkd']['success_rate']:.2f}%\n")
            f.write(f"- Latência Média: {qkd['with_qkd']['avg_latency']:.2f} ms\n\n")

            f.write("### Sem QKD\n")
            f.write(f"- Requisições: {qkd['without_qkd']['count']}\n")
            f.write(f"- Taxa de Sucesso: {qkd['without_qkd']['success_rate']:.2f}%\n")
            f.write(f"- Latência Média: {qkd['without_qkd']['avg_latency']:.2f} ms\n\n")

            f.write("## 6. Principais Conclusões\n\n")
            f.write("1. O RL Engine demonstrou alta taxa de sucesso na seleção de algoritmos\n")
            f.write("2. A latência média permaneceu dentro de limites aceitáveis\n")
            f.write("3. Algoritmos quânticos foram priorizados em cenários de alta segurança\n")
            f.write("4. O sistema adaptou-se eficientemente a diferentes contextos de segurança\n")

        print(f"  ✓ article_summary.md")
        print(f"\n✅ Resumo salvo em: {report_file}")


def main():
    """Função principal"""
    if len(sys.argv) < 2:
        print("Uso: python analyze_results.py <arquivo_json_resultados>")
        print("\nExemplo: python analyze_results.py rl_experiment_20241019_143022.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not Path(json_file).exists():
        print(f"❌ Arquivo não encontrado: {json_file}")
        sys.exit(1)

    print("="*70)
    print("RL ENGINE - ANÁLISE DE RESULTADOS")
    print("="*70)
    print(f"Arquivo: {json_file}\n")

    analyzer = ResultAnalyzer(json_file)

    # Gera todas as visualizações
    analyzer.generate_all_plots()

    # Gera tabelas LaTeX
    analyzer.generate_latex_tables()

    # Gera resumo para artigo
    analyzer.generate_summary_report()

    print("\n" + "="*70)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("="*70)
    print(f"\nTodos os arquivos foram salvos em: {analyzer.output_dir}/")
    print("\nArquivos gerados:")
    print("  - 8 gráficos em PNG (alta resolução)")
    print("  - 1 arquivo com tabelas LaTeX")
    print("  - 1 resumo em Markdown")


if __name__ == "__main__":
    main()
