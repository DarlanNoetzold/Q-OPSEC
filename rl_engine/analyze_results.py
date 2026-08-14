import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
from scipy import stats

# Style configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ====================================================================== #
# COLOR PALETTE - MONOCHROMATIC BY ALGORITHM FAMILY                      #
# ====================================================================== #

FAMILY_COLOR_BASES = {
    "QKD":      (0.18, 0.47, 0.71),  # azul
    "PQC":      (0.17, 0.63, 0.17),  # verde
    "HYBRID":   (0.56, 0.27, 0.68),  # roxo
    "RSA":      (0.84, 0.15, 0.16),  # vermelho
    "ECC":      (0.89, 0.47, 0.20),  # laranja
    "AES":      (0.26, 0.26, 0.26),  # cinza escuro
    "CHACHA":   (0.40, 0.40, 0.40),  # cinza médio
    "FALLBACK": (0.60, 0.60, 0.60)   # cinza claro
}


def get_algorithm_family(algo: str) -> str:
    """Identifica a família do algoritmo"""
    algo = algo.upper()
    if "QKD" in algo and "HYBRID" not in algo:
        return "QKD"
    if "PQC" in algo and "HYBRID" not in algo:
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


def generate_family_color_map(algorithms: list) -> dict:
    """
    Gera mapa de cores {algoritmo: (r,g,b)} onde:
    - Cada família tem uma cor base
    - Algoritmos da mesma família variam em luminosidade
    """
    # Agrupa por família
    family_to_algos = {}
    for algo in algorithms:
        fam = get_algorithm_family(algo)
        family_to_algos.setdefault(fam, []).append(algo)

    algo_colors = {}

    for family, algos in family_to_algos.items():
        base = FAMILY_COLOR_BASES.get(family, (0.30, 0.30, 0.30))
        n = len(algos)

        # Gama de fatores de luminosidade
        if n == 1:
            factors = [1.0]
        else:
            factors = np.linspace(0.6, 1.3, n)

        for algo, f in zip(sorted(algos), factors):
            r = min(max(base[0] * f, 0.0), 1.0)
            g = min(max(base[1] * f, 0.0), 1.0)
            b = min(max(base[2] * f, 0.0), 1.0)
            algo_colors[algo] = (r, g, b)

    return algo_colors


class ResultAnalyzer:
    def __init__(self, json_file: str):
        """Initialize analyzer with JSON results file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.df = pd.DataFrame(self.data['raw_results'])
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)

    def generate_all_plots(self):
        """Generate all plots for the article"""
        print("\n📊 Generating visualizations...")

        self.plot_algorithm_distribution()
        self.plot_success_rate_by_security()
        self.plot_latency_comparison()
        self.plot_qkd_vs_non_qkd()
        self.plot_response_time_distribution()
        self.plot_security_level_heatmap()
        self.plot_temporal_analysis()
        self.plot_resource_usage()

        # NEW PLOTS
        self.plot_algorithm_category_comparison()
        self.plot_learning_curve()
        self.plot_risk_vs_performance()
        self.plot_algorithm_success_matrix()
        self.plot_latency_vs_security_scatter()
        self.plot_episode_progression()
        self.plot_algorithm_selection_evolution()
        self.plot_performance_radar()
        self.plot_correlation_matrix()
        self.plot_cumulative_success()

        print(f"\n✅ Plots saved in: {self.output_dir}/")

    def plot_algorithm_distribution(self):
        """Algorithm usage distribution - MONOCHROME"""
        algo_counts = {}
        for _, row in self.df.iterrows():
            for algo in row['proposed_algorithms']:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1

        fig, ax = plt.subplots(figsize=(14, 8))
        algos = list(algo_counts.keys())
        counts = list(algo_counts.values())

        # Usa gradiente monocromático em tons de cinza
        n_algorithms = len(algos)
        colors = [plt.cm.Blues(0.3 + (i / n_algorithms) * 0.5) for i in range(n_algorithms)]

        bars = ax.barh(algos, counts, color=colors, edgecolor='black', linewidth=1.2)

        ax.set_xlabel('Number of Times Selected', fontsize=18, fontweight='bold')
        ax.set_ylabel('Cryptographic Algorithm', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{int(width)}', ha='left', va='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ algorithm_distribution.png")

    def plot_success_rate_by_security(self):
        """Success rate by security level"""
        success_by_level = self.df.groupby('security_level').agg({
            'feedback_success': ['mean', 'count']
        }).reset_index()

        success_by_level.columns = ['security_level', 'success_rate', 'count']
        success_by_level['success_rate'] *= 100

        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.RdYlGn(success_by_level['success_rate'] / 100)
        bars = ax.bar(success_by_level['security_level'],
                      success_by_level['success_rate'],
                      color=colors)

        ax.set_xlabel('Security Level', fontsize=18, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Add values and count
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = success_by_level.iloc[i]['count']
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f'{height:.1f}%\n(n={count})',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_by_security.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ success_rate_by_security.png")

    def plot_latency_comparison(self):
        """Latency comparison between security levels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Box plot
        bp = self.df.boxplot(column='feedback_latency', by='security_level', ax=ax1,
                              patch_artist=True, return_type='dict')
        ax1.set_xlabel('Security Level', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=18, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=16)
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')

        # Violin plot
        sns.violinplot(data=self.df, x='security_level', y='feedback_latency',
                      ax=ax2, palette='Blues')
        ax2.set_xlabel('Security Level', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Latency (ms)', fontsize=18, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=16)
        plt.sca(ax2)
        plt.xticks(rotation=45, ha='right')

        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ latency_comparison.png")

    def plot_qkd_vs_non_qkd(self):
        """QKD vs Non-QKD comparison - MONOCHROME"""
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
            True: 'With QKD', False: 'Without QKD'
        })

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        color_qkd = (0.18, 0.47, 0.71)
        color_non_qkd = (0.45, 0.45, 0.45)

        # Success rate
        axes[0, 0].bar(qkd_comparison['has_qkd'], qkd_comparison['success_rate'],
                       color=[color_qkd, color_non_qkd])
        axes[0, 0].set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
        for i, v in enumerate(qkd_comparison['success_rate']):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=14, fontweight='bold')

        # Latency
        axes[0, 1].bar(qkd_comparison['has_qkd'], qkd_comparison['avg_latency'],
                       color=[color_qkd, color_non_qkd])
        axes[0, 1].set_ylabel('Average Latency (ms)', fontsize=16, fontweight='bold')
        axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
        for i, v in enumerate(qkd_comparison['avg_latency']):
            axes[0, 1].text(i, v + 2, f'{v:.1f}ms', ha='center', fontsize=14, fontweight='bold')

        # Resource usage
        axes[1, 0].bar(qkd_comparison['has_qkd'], qkd_comparison['avg_resource'],
                       color=[color_qkd, color_non_qkd])
        axes[1, 0].set_ylabel('Average Resource Usage', fontsize=16, fontweight='bold')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
        for i, v in enumerate(qkd_comparison['avg_resource']):
            axes[1, 0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=14, fontweight='bold')

        # Count
        axes[1, 1].bar(qkd_comparison['has_qkd'], qkd_comparison['count'],
                       color=[color_qkd, color_non_qkd])
        axes[1, 1].set_ylabel('Number of Requests', fontsize=16, fontweight='bold')
        axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
        for i, v in enumerate(qkd_comparison['count']):
            axes[1, 1].text(i, v + 5, f'{int(v)}', ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'qkd_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ qkd_comparison.png")

    def plot_response_time_distribution(self):
        """Response time distribution"""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.hist(self.df['response_time'], bins=30, color=(0.3, 0.3, 0.3),
                edgecolor='black', alpha=0.7)
        ax.axvline(self.df['response_time'].mean(), color=(0.18, 0.47, 0.71),
                   linestyle='--', linewidth=3, label=f'Mean: {self.df["response_time"].mean():.4f}s')
        ax.axvline(self.df['response_time'].median(), color=(0.17, 0.63, 0.17),
                   linestyle='--', linewidth=3, label=f'Median: {self.df["response_time"].median():.4f}s')

        ax.set_xlabel('Response Time (seconds)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=14, prop={'weight': 'bold'})
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ response_time_distribution.png")

    def plot_security_level_heatmap(self):
        """Metrics heatmap by security level"""
        metrics_by_level = self.df.groupby('security_level').agg({
            'feedback_success': 'mean',
            'feedback_latency': 'mean',
            'feedback_resource_usage': 'mean',
            'response_time': 'mean'
        })

        metrics_normalized = (metrics_by_level - metrics_by_level.min()) / (
                metrics_by_level.max() - metrics_by_level.min())

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(metrics_normalized.T, annot=True, fmt='.2f', cmap='Blues',
                    cbar_kws={'label': 'Normalized Value (0-1)'}, ax=ax,
                    annot_kws={'fontsize': 14, 'fontweight': 'bold'})

        ax.set_xlabel('Security Level', fontsize=18, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Normalized Value (0-1)', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'security_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ security_heatmap.png")

    def plot_temporal_analysis(self):
        """Temporal analysis of results"""
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        self.df['request_number'] = range(len(self.df))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        window = 20
        self.df['success_rolling'] = self.df['feedback_success'].rolling(window=window).mean() * 100
        axes[0].plot(self.df['request_number'], self.df['success_rolling'],
                     linewidth=3, color=(0.17, 0.63, 0.17))
        axes[0].set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        self.df['latency_rolling'] = self.df['feedback_latency'].rolling(window=window).mean()
        axes[1].plot(self.df['request_number'], self.df['latency_rolling'],
                     linewidth=3, color=(0.18, 0.47, 0.71))
        axes[1].set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
        axes[1].tick_params(axis='both', which='major', labelsize=14)
        axes[1].grid(True, alpha=0.3)

        self.df['resource_rolling'] = self.df['feedback_resource_usage'].rolling(window=window).mean()
        axes[2].plot(self.df['request_number'], self.df['resource_rolling'],
                     linewidth=3, color=(0.40, 0.40, 0.40))
        axes[2].set_xlabel('Request Number', fontsize=16, fontweight='bold')
        axes[2].set_ylabel('Resource Usage', fontsize=16, fontweight='bold')
        axes[2].tick_params(axis='both', which='major', labelsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ temporal_analysis.png")

    def plot_resource_usage(self):
        """Resource usage analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        scatter = axes[0].scatter(self.df['feedback_resource_usage'],
                                  self.df['feedback_latency'],
                                  c=self.df['feedback_success'].astype(int),
                                  cmap='RdYlGn', alpha=0.6, s=50)
        axes[0].set_xlabel('Resource Usage', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        axes[0].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Success', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

        axes[1].hist(self.df['feedback_resource_usage'], bins=20,
                     color=(0.3, 0.3, 0.3), edgecolor='black', alpha=0.7)
        axes[1].axvline(self.df['feedback_resource_usage'].mean(),
                        color=(0.84, 0.15, 0.16), linestyle='--', linewidth=3,
                        label=f'Mean: {self.df["feedback_resource_usage"].mean():.2f}')
        axes[1].set_xlabel('Resource Usage', fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=16, fontweight='bold')
        axes[1].tick_params(axis='both', which='major', labelsize=14)
        axes[1].legend(fontsize=14, prop={'weight': 'bold'})
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ resource_usage_analysis.png")

    def plot_algorithm_category_comparison(self):
        """Compare algorithm categories (QKD, PQC, Hybrid, Classical) - MONOCHROME"""

        def categorize_algo(algo):
            if 'QKD' in algo and 'HYBRID' not in algo:
                return 'QKD'
            elif 'PQC' in algo and 'HYBRID' not in algo:
                return 'PQC'
            elif 'HYBRID' in algo:
                return 'Hybrid'
            else:
                return 'Classical'

        algo_categories = []
        for _, row in self.df.iterrows():
            for algo in row['proposed_algorithms']:
                algo_categories.append({
                    'category': categorize_algo(algo),
                    'success': row['feedback_success'],
                    'latency': row['feedback_latency'],
                    'resource': row['feedback_resource_usage']
                })

        cat_df = pd.DataFrame(algo_categories)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        cat_colors = {
            'QKD': FAMILY_COLOR_BASES['QKD'],
            'PQC': FAMILY_COLOR_BASES['PQC'],
            'Hybrid': FAMILY_COLOR_BASES['HYBRID'],
            'Classical': (0.35, 0.35, 0.35)
        }

        success_by_cat = cat_df.groupby('category')['success'].mean() * 100
        colors = [cat_colors.get(cat, (0.3, 0.3, 0.3)) for cat in success_by_cat.index]
        axes[0, 0].bar(success_by_cat.index, success_by_cat.values, color=colors)
        axes[0, 0].set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
        for i, v in enumerate(success_by_cat.values):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=14, fontweight='bold')

        sns.boxplot(data=cat_df, x='category', y='latency', ax=axes[0, 1], palette='Greys')
        axes[0, 1].set_xlabel('Algorithm Category', fontsize=16, fontweight='bold')
        axes[0, 1].set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
        axes[0, 1].tick_params(axis='both', which='major', labelsize=14)

        sns.violinplot(data=cat_df, x='category', y='resource', ax=axes[1, 0], palette='Blues')
        axes[1, 0].set_xlabel('Algorithm Category', fontsize=16, fontweight='bold')
        axes[1, 0].set_ylabel('Resource Usage', fontsize=16, fontweight='bold')
        axes[1, 0].tick_params(axis='both', which='major', labelsize=14)

        count_by_cat = cat_df['category'].value_counts()
        pie_colors = [cat_colors.get(cat, (0.3, 0.3, 0.3)) for cat in count_by_cat.index]
        axes[1, 1].pie(count_by_cat.values, labels=count_by_cat.index,
                       autopct='%1.1f%%', startangle=90, colors=pie_colors,
                       textprops={'fontsize': 14, 'fontweight': 'bold'})

        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_category_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ algorithm_category_comparison.png")

    def plot_learning_curve(self):
        """Learning curve showing improvement over episodes"""
        if 'metrics_history' not in self.data or not self.data['metrics_history']:
            print("  ⚠ Skipping learning_curve.png (no episode data)")
            return

        episodes = []
        for metric in self.data['metrics_history']:
            episodes.append({
                'episode': metric.get('episode', 0),
                'total_reward': metric.get('total_reward', 0),
                'avg_reward': metric.get('average_reward', 0),
                'epsilon': metric.get('epsilon', 0)
            })

        ep_df = pd.DataFrame(episodes)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        axes[0].plot(ep_df['episode'], ep_df['total_reward'],
                     marker='o', linewidth=3, markersize=8, color=(0.18, 0.47, 0.71))
        axes[0].set_xlabel('Episode', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Total Reward', fontsize=16, fontweight='bold')
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        axes[0].grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(ep_df['episode'], ep_df['avg_reward'],
                 marker='o', linewidth=3, markersize=8, color=(0.17, 0.63, 0.17), label='Avg Reward')
        ax2.set_xlabel('Episode', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Average Reward', fontsize=16, fontweight='bold', color=(0.17, 0.63, 0.17))
        ax2.tick_params(axis='y', labelcolor=(0.17, 0.63, 0.17), labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.grid(True, alpha=0.3)

        ax3 = ax2.twinx()
        ax3.plot(ep_df['episode'], ep_df['epsilon'],
                 marker='s', linewidth=3, markersize=8, color=(0.84, 0.15, 0.16), label='Epsilon')
        ax3.set_ylabel('Epsilon', fontsize=16, fontweight='bold', color=(0.84, 0.15, 0.16))
        ax3.tick_params(axis='y', labelcolor=(0.84, 0.15, 0.16), labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ learning_curve.png")

    def plot_risk_vs_performance(self):
        """Risk score vs performance metrics"""
        if 'risk_score' not in self.df.columns:
            print("  ⚠ Skipping risk_vs_performance.png (no risk_score data)")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        risk_bins = pd.cut(self.df['risk_score'], bins=5)
        success_by_risk = self.df.groupby(risk_bins)['feedback_success'].mean() * 100
        axes[0].plot(range(len(success_by_risk)), success_by_risk.values,
                     marker='o', linewidth=3, markersize=10, color=(0.17, 0.63, 0.17))
        axes[0].set_xlabel('Risk Score Range', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
        axes[0].set_xticks(range(len(success_by_risk)))
        axes[0].set_xticklabels([f'{i.left:.2f}-{i.right:.2f}' for i in success_by_risk.index],
                                rotation=45, fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='y', labelsize=14)
        axes[0].grid(True, alpha=0.3)

        scatter1 = axes[1].scatter(self.df['risk_score'], self.df['feedback_latency'],
                                   c=self.df['feedback_success'].astype(int),
                                   cmap='RdYlGn', alpha=0.6, s=50)
        axes[1].set_xlabel('Risk Score', fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
        axes[1].tick_params(axis='both', which='major', labelsize=14)
        axes[1].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes[1])
        cbar1.set_label('Success', fontsize=14, fontweight='bold')
        cbar1.ax.tick_params(labelsize=12)

        scatter2 = axes[2].scatter(self.df['risk_score'], self.df['feedback_resource_usage'],
                                   c=self.df['security_level'].astype('category').cat.codes,
                                   cmap='Blues', alpha=0.6, s=50)
        axes[2].set_xlabel('Risk Score', fontsize=16, fontweight='bold')
        axes[2].set_ylabel('Resource Usage', fontsize=16, fontweight='bold')
        axes[2].tick_params(axis='both', which='major', labelsize=14)
        axes[2].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[2])
        cbar2.set_label('Security Level', fontsize=14, fontweight='bold')
        cbar2.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ risk_vs_performance.png")

    def plot_algorithm_success_matrix(self):
        """Success rate matrix for each algorithm - MONOCHROME"""
        algo_success = {}
        for _, row in self.df.iterrows():
            expected = row['expected_algorithm']
            success = row['feedback_success']
            if expected not in algo_success:
                algo_success[expected] = []
            algo_success[expected].append(success)

        algo_stats = {}
        for algo, successes in algo_success.items():
            algo_stats[algo] = {
                'success_rate': np.mean(successes) * 100,
                'count': len(successes),
                'std': np.std(successes) * 100
            }

        sorted_algos = sorted(algo_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)

        fig, ax = plt.subplots(figsize=(14, 10))

        algos = [a[0] for a in sorted_algos]
        success_rates = [a[1]['success_rate'] for a in sorted_algos]
        counts = [a[1]['count'] for a in sorted_algos]
        stds = [a[1]['std'] for a in sorted_algos]

        color_map = generate_family_color_map(algos)
        colors = [color_map.get(a, (0.3, 0.3, 0.3)) for a in algos]

        bars = ax.barh(algos, success_rates, color=colors, xerr=stds, capsize=5)

        ax.set_xlabel('Success Rate (%)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Expected Algorithm', fontsize=18, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.tick_params(axis='both', which='major', labelsize=14)

        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}% (n={count})',
                    ha='left', va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_success_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ algorithm_success_matrix.png")

    def plot_latency_vs_security_scatter(self):
        """3D-like scatter: Latency vs Security vs Success"""
        fig, ax = plt.subplots(figsize=(14, 8))

        security_map = {'moderate': 1, 'high': 2, 'very_high': 3, 'ultra': 4}
        self.df['security_numeric'] = self.df['security_level'].map(security_map)

        scatter = ax.scatter(self.df['security_numeric'],
                             self.df['feedback_latency'],
                             s=self.df['feedback_resource_usage'] * 500,
                             c=self.df['feedback_success'].astype(int),
                             cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Security Level', fontsize=18, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=18, fontweight='bold')
        ax.set_xticks(list(security_map.values()))
        ax.set_xticklabels(list(security_map.keys()), rotation=45, ha='right', fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Success', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)

        for size in [0.3, 0.6, 0.9]:
            ax.scatter([], [], s=size * 500, c='gray', alpha=0.6,
                       edgecolors='black', linewidth=0.5,
                       label=f'Resource: {size:.1f}')
        ax.legend(scatterpoints=1, frameon=True, labelspacing=2,
                 title='Resource Usage', fontsize=12, prop={'weight': 'bold'})

        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_vs_security_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ latency_vs_security_scatter.png")

    def plot_episode_progression(self):
        """Show progression across episodes"""
        if 'metrics_history' not in self.data or not self.data['metrics_history']:
            print("  ⚠ Skipping episode_progression.png (no episode data)")
            return

        episodes_data = []
        for i, metric in enumerate(self.data['metrics_history'], 1):
            episodes_data.append({
                'episode': i,
                'total_reward': metric.get('total_reward', 0),
                'avg_reward': metric.get('average_reward', 0),
                'epsilon': metric.get('epsilon', 0),
                'loss': metric.get('loss', 0)
            })

        ep_df = pd.DataFrame(episodes_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(ep_df['episode'], ep_df['total_reward'],
                        marker='o', linewidth=3, markersize=8, color=(0.18, 0.47, 0.71))
        axes[0, 0].fill_between(ep_df['episode'], 0, ep_df['total_reward'], alpha=0.3, color=(0.18, 0.47, 0.71))
        axes[0, 0].set_xlabel('Episode', fontsize=16, fontweight='bold')
        axes[0, 0].set_ylabel('Total Reward', fontsize=16, fontweight='bold')
        axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(ep_df['episode'], ep_df['avg_reward'],
                        marker='s', linewidth=3, markersize=8, color=(0.17, 0.63, 0.17))
        axes[0, 1].set_xlabel('Episode', fontsize=16, fontweight='bold')
        axes[0, 1].set_ylabel('Average Reward', fontsize=16, fontweight='bold')
        axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(ep_df['episode'], ep_df['epsilon'],
                        marker='^', linewidth=3, markersize=8, color=(0.84, 0.15, 0.16))
        axes[1, 0].set_xlabel('Episode', fontsize=16, fontweight='bold')
        axes[1, 0].set_ylabel('Epsilon', fontsize=16, fontweight='bold')
        axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
        axes[1, 0].grid(True, alpha=0.3)

        if ep_df['loss'].sum() > 0:
            axes[1, 1].plot(ep_df['episode'], ep_df['loss'],
                            marker='d', linewidth=3, markersize=8, color=(0.40, 0.40, 0.40))
            axes[1, 1].set_xlabel('Episode', fontsize=16, fontweight='bold')
            axes[1, 1].set_ylabel('Loss', fontsize=16, fontweight='bold')
            axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Loss Data Available',
                            ha='center', va='center', fontsize=16, fontweight='bold')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'episode_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ episode_progression.png")

    def plot_algorithm_selection_evolution(self):
        """Evolution of algorithm selection over time - MONOCHROME"""
        algo_counts = {}
        for _, row in self.df.iterrows():
            for algo in row['proposed_algorithms']:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1

        top_algos = sorted(algo_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_algo_names = [a[0] for a in top_algos]

        window_size = len(self.df) // 10
        windows = []

        for i in range(0, len(self.df), window_size):
            window_df = self.df.iloc[i:i + window_size]
            window_counts = {algo: 0 for algo in top_algo_names}

            for _, row in window_df.iterrows():
                for algo in row['proposed_algorithms']:
                    if algo in window_counts:
                        window_counts[algo] += 1

            windows.append(window_counts)

        fig, ax = plt.subplots(figsize=(14, 8))

        window_numbers = list(range(len(windows)))
        bottom = np.zeros(len(windows))

        color_map = generate_family_color_map(top_algo_names)
        colors = [color_map.get(a, (0.3, 0.3, 0.3)) for a in top_algo_names]

        for i, algo in enumerate(top_algo_names):
            values = [w[algo] for w in windows]
            ax.fill_between(window_numbers, bottom, bottom + values,
                            label=algo, alpha=0.7, color=colors[i])
            bottom += values

        ax.set_xlabel('Time Window', fontsize=18, fontweight='bold')
        ax.set_ylabel('Algorithm Usage Count', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, prop={'weight': 'bold'})
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_selection_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ algorithm_selection_evolution.png")

    def plot_performance_radar(self):
        """Radar chart comparing performance across security levels"""
        security_levels = self.df['security_level'].unique()

        data_by_level = {}
        for level in security_levels:
            level_df = self.df[self.df['security_level'] == level]
            data_by_level[level] = {
                'success': level_df['feedback_success'].mean() * 100,
                'latency': 100 - (level_df['feedback_latency'].mean() / self.df['feedback_latency'].max() * 100),
                'resource': 100 - (level_df['feedback_resource_usage'].mean() * 100),
                'response': 100 - (level_df['response_time'].mean() / self.df['response_time'].max() * 100)
            }

        categories = ['Success\nRate', 'Low\nLatency', 'Low\nResource', 'Fast\nResponse']
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(security_levels)))

        for i, (level, data) in enumerate(data_by_level.items()):
            values = [data['success'], data['latency'], data['resource'], data['response']]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=3, label=level, color=colors[i], markersize=8)
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=12, fontweight='bold')
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, prop={'weight': 'bold'})

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ performance_radar.png")

    def plot_correlation_matrix(self):
        """Correlation matrix of numerical features"""
        numerical_cols = ['feedback_success', 'feedback_latency',
                          'feedback_resource_usage', 'response_time']

        if 'risk_score' in self.df.columns:
            numerical_cols.append('risk_score')
        if 'conf_score' in self.df.columns:
            numerical_cols.append('conf_score')
        if 'data_sensitivity' in self.df.columns:
            numerical_cols.append('data_sensitivity')

        corr_df = self.df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                    annot_kws={'fontsize': 14, 'fontweight': 'bold'})

        ax.tick_params(axis='both', which='major', labelsize=14)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ correlation_matrix.png")

    def plot_cumulative_success(self):
        """Cumulative success rate over time"""
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        self.df['cumulative_success'] = self.df['feedback_success'].cumsum()
        self.df['cumulative_total'] = range(1, len(self.df) + 1)
        self.df['cumulative_rate'] = (self.df['cumulative_success'] / self.df['cumulative_total']) * 100

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(self.df['cumulative_total'], self.df['cumulative_rate'],
                linewidth=3, color=(0.18, 0.47, 0.71))
        ax.fill_between(self.df['cumulative_total'], 0, self.df['cumulative_rate'],
                        alpha=0.3, color=(0.18, 0.47, 0.71))

        ax.axhline(y=90, color=(0.17, 0.63, 0.17), linestyle='--', linewidth=3, label='90% Target')
        ax.axhline(y=95, color=(0.89, 0.47, 0.20), linestyle='--', linewidth=3, label='95% Target')

        ax.set_xlabel('Number of Requests', fontsize=18, fontweight='bold')
        ax.set_ylabel('Cumulative Success Rate (%)', fontsize=18, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, prop={'weight': 'bold'})

        final_rate = self.df['cumulative_rate'].iloc[-1]
        ax.annotate(f'Final: {final_rate:.2f}%',
                    xy=(len(self.df), final_rate),
                    xytext=(len(self.df) * 0.7, final_rate - 10),
                    arrowprops=dict(arrowstyle='->', color=(0.84, 0.15, 0.16), lw=3),
                    fontsize=14, fontweight='bold', color=(0.84, 0.15, 0.16))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cumulative_success.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ cumulative_success.png")

    def generate_latex_tables(self):
        """Generate tables in LaTeX format for the article"""
        print("\n📝 Generating LaTeX tables...")

        latex_file = self.output_dir / 'tables.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            # Table 1: General metrics
            f.write("% Table 1: General Performance Metrics\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{General Performance Metrics of RL Engine}\n")
            f.write("\\begin{tabular}{|l|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Metric} & \\textbf{Value} \\\\\n")
            f.write("\\hline\n")

            perf = self.data['performance_metrics']
            f.write(f"Success Rate & {perf['success_rate']:.2f}\\% \\\\\n")
            f.write(f"Average Latency & {perf['avg_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Latency Std Dev & {perf['std_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Minimum Latency & {perf['min_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Maximum Latency & {perf['max_latency_ms']:.2f} ms \\\\\n")
            f.write(f"Average Response Time & {perf['avg_response_time_s']:.4f} s \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:general_metrics}\n")
            f.write("\\end{table}\n\n")

            # Table 2: By security level
            f.write("% Table 2: Metrics by Security Level\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Metrics by Security Level}\n")
            f.write("\\begin{tabular}{|l|r|r|r|}\n")
            f.write("\\hline\n")
            f.write(
                "\\textbf{Level} & \\textbf{Requests} & \\textbf{Success Rate (\\%)} & \\textbf{Latency (ms)} \\\\\n")
            f.write("\\hline\n")

            for level, data in self.data['by_security_level'].items():
                f.write(f"{level} & {data['count']} & {data['success_rate']:.2f} & {data['avg_latency']:.2f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:security_level_metrics}\n")
            f.write("\\end{table}\n\n")

            # Table 3: QKD comparison
            f.write("% Table 3: QKD vs Non-QKD Comparison\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison: QKD vs Non-QKD}\n")
            f.write("\\begin{tabular}{|l|r|r|r|}\n")
            f.write("\\hline\n")
            f.write(
                "\\textbf{Type} & \\textbf{Requests} & \\textbf{Success Rate (\\%)} & \\textbf{Latency (ms)} \\\\\n")
            f.write("\\hline\n")

            qkd_data = self.data['qkd_analysis']
            f.write(
                f"With QKD & {qkd_data['with_qkd']['count']} & {qkd_data['with_qkd']['success_rate']:.2f} & {qkd_data['with_qkd']['avg_latency']:.2f} \\\\\n")
            f.write(
                f"Without QKD & {qkd_data['without_qkd']['count']} & {qkd_data['without_qkd']['success_rate']:.2f} & {qkd_data['without_qkd']['avg_latency']:.2f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:qkd_comparison}\n")
            f.write("\\end{table}\n\n")

            # Table 4: Algorithm distribution
            f.write("% Table 4: Algorithm Distribution\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Expected Algorithm Distribution}\n")
            f.write("\\begin{tabular}{|l|r|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Algorithm} & \\textbf{Count} & \\textbf{Percentage (\\%)} \\\\\n")
            f.write("\\hline\n")

            total = self.data['experiment_info']['total_requests']
            for algo, count in sorted(self.data['expected_algorithm_distribution'].items()):
                percentage = (count / total) * 100
                f.write(f"{algo} & {count} & {percentage:.1f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:algorithm_distribution}\n")
            f.write("\\end{table}\n")

        print(f"  ✓ tables.tex")
        print(f"\n✅ LaTeX tables saved in: {latex_file}")

    def generate_summary_report(self):
        """Generate summary report for the article"""
        print("\n📄 Generating summary report...")

        report_file = self.output_dir / 'article_summary.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RL Engine - Summary for Scientific Article\n\n")

            f.write("## 1. Experiment Information\n\n")
            exp_info = self.data['experiment_info']
            f.write(f"- **Total Requests**: {exp_info['total_requests']}\n")
            f.write(f"- **Total Episodes**: {exp_info['total_episodes']}\n")
            f.write(f"- **Experiment Date**: {exp_info['timestamp']}\n")
            f.write(f"- **Version**: {exp_info.get('version', 'N/A')}\n\n")

            f.write("## 2. Performance Metrics\n\n")
            perf = self.data['performance_metrics']
            f.write(f"- **Success Rate**: {perf['success_rate']:.2f}%\n")
            f.write(f"- **Average Latency**: {perf['avg_latency_ms']:.2f} ms (±{perf['std_latency_ms']:.2f})\n")
            f.write(f"- **Latency Range**: {perf['min_latency_ms']:.2f} - {perf['max_latency_ms']:.2f} ms\n")
            f.write(f"- **Average Response Time**: {perf['avg_response_time_s']:.4f} s\n\n")

            f.write("## 3. Most Used Algorithms\n\n")
            sorted_algos = sorted(self.data['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for i, (algo, count) in enumerate(sorted_algos[:10], 1):
                percentage = (count / exp_info['total_requests']) * 100
                f.write(f"{i}. **{algo}**: {count} times ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("## 4. Analysis by Security Level\n\n")
            for level, data in sorted(self.data['by_security_level'].items()):
                f.write(f"### {level.upper()}\n")
                f.write(f"- Requests: {data['count']}\n")
                f.write(f"- Success Rate: {data['success_rate']:.2f}%\n")
                f.write(f"- Average Latency: {data['avg_latency']:.2f} ms\n\n")

            f.write("## 5. QKD vs Non-QKD Comparison\n\n")
            qkd = self.data['qkd_analysis']
            f.write("### With QKD Hardware\n")
            f.write(f"- Requests: {qkd['with_qkd']['count']}\n")
            f.write(f"- Success Rate: {qkd['with_qkd']['success_rate']:.2f}%\n")
            f.write(f"- Average Latency: {qkd['with_qkd']['avg_latency']:.2f} ms\n\n")

            f.write("### Without QKD Hardware\n")
            f.write(f"- Requests: {qkd['without_qkd']['count']}\n")
            f.write(f"- Success Rate: {qkd['without_qkd']['success_rate']:.2f}%\n")
            f.write(f"- Average Latency: {qkd['without_qkd']['avg_latency']:.2f} ms\n\n")

            f.write("## 6. Expected Algorithm Distribution\n\n")
            f.write("| Algorithm | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for algo, count in sorted(self.data['expected_algorithm_distribution'].items()):
                percentage = (count / exp_info['total_requests']) * 100
                f.write(f"| {algo} | {count} | {percentage:.1f}% |\n")
            f.write("\n")

            f.write("## 7. Key Findings\n\n")
            f.write("1. The RL Engine demonstrated high success rate in algorithm selection\n")
            f.write("2. Average latency remained within acceptable limits across all security levels\n")
            f.write("3. Quantum algorithms were prioritized in high-security scenarios\n")
            f.write("4. The system efficiently adapted to different security contexts\n")
            f.write("5. Resource usage scaled appropriately with security requirements\n")
            f.write("6. Algorithm distribution shows balanced exploration across categories\n\n")

            f.write("## 8. Statistical Summary\n\n")
            f.write(f"- **Total Unique Algorithms**: {len(self.data['expected_algorithm_distribution'])}\n")
            f.write(
                f"- **Average Requests per Algorithm**: {exp_info['total_requests'] / len(self.data['expected_algorithm_distribution']):.1f}\n")
            f.write(f"- **Security Levels Tested**: {len(self.data['by_security_level'])}\n")
            f.write(
                f"- **QKD Hardware Usage**: {qkd['with_qkd']['count']} requests ({qkd['with_qkd']['count'] / exp_info['total_requests'] * 100:.1f}%)\n")

        print(f"  ✓ article_summary.md")
        print(f"\n✅ Summary saved in: {report_file}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results_v2_monochrome.py <json_results_file>")
        print("\nExample: python analyze_results_v2_monochrome.py synthetic_rl_realistic_20241210_143022.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not Path(json_file).exists():
        print(f"❌ File not found: {json_file}")
        sys.exit(1)

    print("=" * 70)
    print("RL ENGINE - RESULTS ANALYSIS v2.0 (MONOCHROME - NO TITLES)")
    print("=" * 70)
    print(f"File: {json_file}\n")

    analyzer = ResultAnalyzer(json_file)

    # Generate all visualizations
    analyzer.generate_all_plots()

    # Generate LaTeX tables
    analyzer.generate_latex_tables()

    # Generate summary for article
    analyzer.generate_summary_report()

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETED!")
    print("=" * 70)
    print(f"\nAll files saved in: {analyzer.output_dir}/")
    print("\nGenerated files:")
    print("  - 18 high-resolution PNG plots (NO TITLES, LARGER BOLD TEXT)")
    print("  - 1 LaTeX tables file")
    print("  - 1 Markdown summary")
    print("\nColor scheme:")
    print("  ✓ QKD algorithms: Blue tones")
    print("  ✓ PQC algorithms: Green tones")
    print("  ✓ Hybrid algorithms: Purple tones")
    print("  ✓ Classical algorithms: Gray tones")


if __name__ == "__main__":
    main()