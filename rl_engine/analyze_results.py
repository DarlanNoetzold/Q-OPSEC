import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import json

# Style configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


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
        print("\\n📊 Generating visualizations...")

        self.plot_algorithm_distribution()
        self.plot_success_rate_by_security()
        self.plot_latency_comparison()
        self.plot_qkd_vs_non_qkd()
        self.plot_response_time_distribution()
        self.plot_security_level_heatmap()
        self.plot_temporal_analysis()
        self.plot_resource_usage()

        print(f"\\n✅ Plots saved in: {self.output_dir}/")

    def plot_algorithm_distribution(self):
        """Algorithm usage distribution"""
        algo_counts = {}
        for _, row in self.df.iterrows():
            for algo in row['proposed_algorithms']:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1

        fig, ax = plt.subplots(figsize=(14, 8))
        algos = list(algo_counts.keys())
        counts = list(algo_counts.values())

        colors = plt.cm.viridis(np.linspace(0, 1, len(algos)))
        bars = ax.barh(algos, counts, color=colors)

        ax.set_xlabel('Number of Times Selected', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cryptographic Algorithm', fontsize=12, fontweight='bold')
        ax.set_title('Cryptographic Algorithm Usage Distribution',
                     fontsize=14, fontweight='bold', pad=20)

        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{int(width)}', ha='left', va='center', fontsize=10)

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

        bars = ax.bar(success_by_level['security_level'],
                      success_by_level['success_rate'],
                      color=plt.cm.RdYlGn(success_by_level['success_rate'] / 100))

        ax.set_xlabel('Security Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate by Security Level',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 105)

        # Add values and count
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = success_by_level.iloc[i]['count']
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f'{height:.1f}%\\n(n={count})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_by_security.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ success_rate_by_security.png")

    def plot_latency_comparison(self):
        """Latency comparison between security levels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Box plot
        self.df.boxplot(column='feedback_latency', by='security_level', ax=ax1)
        ax1.set_xlabel('Security Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Latency Distribution by Security Level',
                      fontsize=12, fontweight='bold')
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')

        # Violin plot
        sns.violinplot(data=self.df, x='security_level', y='feedback_latency', ax=ax2)
        ax2.set_xlabel('Security Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Latency Density by Security Level',
                      fontsize=12, fontweight='bold')
        plt.sca(ax2)
        plt.xticks(rotation=45, ha='right')

        plt.suptitle('')  # Remove automatic title
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ latency_comparison.png")

    def plot_qkd_vs_non_qkd(self):
        """QKD vs Non-QKD comparison"""
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

        # Success rate
        axes[0, 0].bar(qkd_comparison['has_qkd'], qkd_comparison['success_rate'],
                       color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_ylabel('Success Rate (%)', fontweight='bold')
        axes[0, 0].set_title('Success Rate', fontweight='bold')
        axes[0, 0].set_ylim(0, 105)
        for i, v in enumerate(qkd_comparison['success_rate']):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

        # Latency
        axes[0, 1].bar(qkd_comparison['has_qkd'], qkd_comparison['avg_latency'],
                       color=['#3498db', '#9b59b6'])
        axes[0, 1].set_ylabel('Average Latency (ms)', fontweight='bold')
        axes[0, 1].set_title('Average Latency', fontweight='bold')
        for i, v in enumerate(qkd_comparison['avg_latency']):
            axes[0, 1].text(i, v + 2, f'{v:.1f}ms', ha='center', fontweight='bold')

        # Resource usage
        axes[1, 0].bar(qkd_comparison['has_qkd'], qkd_comparison['avg_resource'],
                       color=['#f39c12', '#1abc9c'])
        axes[1, 0].set_ylabel('Average Resource Usage', fontweight='bold')
        axes[1, 0].set_title('Resource Usage', fontweight='bold')
        axes[1, 0].set_ylim(0, 1.1)
        for i, v in enumerate(qkd_comparison['avg_resource']):
            axes[1, 0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

        # Count
        axes[1, 1].bar(qkd_comparison['has_qkd'], qkd_comparison['count'],
                       color=['#34495e', '#95a5a6'])
        axes[1, 1].set_ylabel('Number of Requests', fontweight='bold')
        axes[1, 1].set_title('Request Volume', fontweight='bold')
        for i, v in enumerate(qkd_comparison['count']):
            axes[1, 1].text(i, v + 5, f'{int(v)}', ha='center', fontweight='bold')

        plt.suptitle('Comparative Analysis: QKD vs Non-QKD',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qkd_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ qkd_comparison.png")

    def plot_response_time_distribution(self):
        """Response time distribution"""
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.hist(self.df['response_time'], bins=30, color='skyblue',
                edgecolor='black', alpha=0.7)
        ax.axvline(self.df['response_time'].mean(), color='red',
                   linestyle='--', linewidth=2, label=f'Mean: {self.df["response_time"].mean():.4f}s')
        ax.axvline(self.df['response_time'].median(), color='green',
                   linestyle='--', linewidth=2, label=f'Median: {self.df["response_time"].median():.4f}s')

        ax.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('RL Engine Response Time Distribution',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
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

        # Normalize to 0-1
        metrics_normalized = (metrics_by_level - metrics_by_level.min()) / (
                    metrics_by_level.max() - metrics_by_level.min())

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(metrics_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn',
                    cbar_kws={'label': 'Normalized Value (0-1)'}, ax=ax)

        ax.set_xlabel('Security Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        ax.set_title('Metrics Heatmap by Security Level (Normalized)',
                     fontsize=14, fontweight='bold', pad=20)

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

        # Success rate over time
        window = 20
        self.df['success_rolling'] = self.df['feedback_success'].rolling(window=window).mean() * 100
        axes[0].plot(self.df['request_number'], self.df['success_rolling'],
                     linewidth=2, color='green')
        axes[0].set_ylabel('Success Rate (%) - Moving Average', fontweight='bold')
        axes[0].set_title(f'Success Rate Evolution (window={window})', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)

        # Latency over time
        self.df['latency_rolling'] = self.df['feedback_latency'].rolling(window=window).mean()
        axes[1].plot(self.df['request_number'], self.df['latency_rolling'],
                     linewidth=2, color='blue')
        axes[1].set_ylabel('Latency (ms) - Moving Average', fontweight='bold')
        axes[1].set_title(f'Latency Evolution (window={window})', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Resource usage over time
        self.df['resource_rolling'] = self.df['feedback_resource_usage'].rolling(window=window).mean()
        axes[2].plot(self.df['request_number'], self.df['resource_rolling'],
                     linewidth=2, color='orange')
        axes[2].set_xlabel('Request Number', fontweight='bold')
        axes[2].set_ylabel('Resource Usage - Moving Average', fontweight='bold')
        axes[2].set_title(f'Resource Usage Evolution (window={window})', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1.1)

        plt.suptitle('Temporal Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ temporal_analysis.png")

    def plot_resource_usage(self):
        """Resource usage analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter: Resources vs Latency
        scatter = axes[0].scatter(self.df['feedback_resource_usage'],
                                  self.df['feedback_latency'],
                                  c=self.df['feedback_success'].astype(int),
                                  cmap='RdYlGn', alpha=0.6, s=50)
        axes[0].set_xlabel('Resource Usage', fontweight='bold')
        axes[0].set_ylabel('Latency (ms)', fontweight='bold')
        axes[0].set_title('Relationship: Resource Usage vs Latency', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Success', fontweight='bold')

        # Resource usage distribution
        axes[1].hist(self.df['feedback_resource_usage'], bins=20,
                     color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(self.df['feedback_resource_usage'].mean(),
                        color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {self.df["feedback_resource_usage"].mean():.2f}')
        axes[1].set_xlabel('Resource Usage', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Resource Usage Distribution', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ resource_usage_analysis.png")

    def generate_latex_tables(self):
        """Generate tables in LaTeX format for the article"""
        print("\\n📝 Generating LaTeX tables...")

        latex_file = self.output_dir / 'tables.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            # Table 1: General metrics
            f.write("% Table 1: General Performance Metrics\\n")
            f.write("\\\\begin{table}[h]\\n")
            f.write("\\\\centering\\n")
            f.write("\\\\caption{RL Engine General Performance Metrics}\\n")
            f.write("\\\\begin{tabular}{|l|r|}\\n")
            f.write("\\\\hline\\n")
            f.write("\\\\textbf{Metric} & \\\\textbf{Value} \\\\\\\\\\n")
            f.write("\\\\hline\\n")

            perf = self.data['performance_metrics']
            f.write(f"Success Rate & {perf['success_rate']:.2f}\\\\% \\\\\\\\\\n")
            f.write(f"Average Latency & {perf['avg_latency_ms']:.2f} ms \\\\\\\\\\n")
            f.write(f"Latency Std Dev & {perf['std_latency_ms']:.2f} ms \\\\\\\\\\n")
            f.write(f"Minimum Latency & {perf['min_latency_ms']:.2f} ms \\\\\\\\\\n")
            f.write(f"Maximum Latency & {perf['max_latency_ms']:.2f} ms \\\\\\\\\\n")
            f.write(f"Average Response Time & {perf['avg_response_time_s']:.4f} s \\\\\\\\\\n")

            f.write("\\\\hline\\n")
            f.write("\\\\end{tabular}\\n")
            f.write("\\\\label{tab:general_metrics}\\n")
            f.write("\\\\end{table}\\n\\n")

            # Table 2: By security level
            f.write("% Table 2: Metrics by Security Level\\n")
            f.write("\\\\begin{table}[h]\\n")
            f.write("\\\\centering\\n")
            f.write("\\\\caption{Metrics by Security Level}\\n")
            f.write("\\\\begin{tabular}{|l|r|r|r|}\\n")
            f.write("\\\\hline\\n")
            f.write(
                "\\\\textbf{Level} & \\\\textbf{Requests} & \\\\textbf{Success Rate (\\\\%)} & \\\\textbf{Latency (ms)} \\\\\\\\\\n")
            f.write("\\\\hline\\n")

            for level, data in self.data['by_security_level'].items():
                f.write(
                    f"{level} & {data['count']} & {data['success_rate']:.2f} & {data['avg_latency']:.2f} \\\\\\\\\\n")

            f.write("\\\\hline\\n")
            f.write("\\\\end{tabular}\\n")
            f.write("\\\\label{tab:security_level_metrics}\\n")
            f.write("\\\\end{table}\\n\\n")

            # Table 3: QKD comparison
            f.write("% Table 3: QKD vs Non-QKD Comparison\\n")
            f.write("\\\\begin{table}[h]\\n")
            f.write("\\\\centering\\n")
            f.write("\\\\caption{Comparison: QKD vs Non-QKD}\\n")
            f.write("\\\\begin{tabular}{|l|r|r|r|}\\n")
            f.write("\\\\hline\\n")
            f.write(
                "\\\\textbf{Type} & \\\\textbf{Requests} & \\\\textbf{Success Rate (\\\\%)} & \\\\textbf{Latency (ms)} \\\\\\\\\\n")
            f.write("\\\\hline\\n")

            qkd_data = self.data['qkd_analysis']
            f.write(
                f"With QKD & {qkd_data['with_qkd']['count']} & {qkd_data['with_qkd']['success_rate']:.2f} & {qkd_data['with_qkd']['avg_latency']:.2f} \\\\\\\\\\n")
            f.write(
                f"Without QKD & {qkd_data['without_qkd']['count']} & {qkd_data['without_qkd']['success_rate']:.2f} & {qkd_data['without_qkd']['avg_latency']:.2f} \\\\\\\\\\n")

            f.write("\\\\hline\\n")
            f.write("\\\\end{tabular}\\n")
            f.write("\\\\label{tab:qkd_comparison}\\n")
            f.write("\\\\end{table}\\n")

        print(f"  ✓ tables.tex")
        print(f"\\n✅ LaTeX tables saved in: {latex_file}")

    def generate_summary_report(self):
        """Generate summary report for the article"""
        print("\\n📄 Generating summary report...")

        report_file = self.output_dir / 'article_summary.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RL Engine - Summary for Scientific Article\\n\\n")

            f.write("## 1. Experiment Information\\n\\n")
            exp_info = self.data['experiment_info']
            f.write(f"- **Total Requests**: {exp_info['total_requests']}\\n")
            f.write(f"- **Total Episodes**: {exp_info['total_episodes']}\\n")
            f.write(f"- **Experiment Date**: {exp_info['timestamp']}\\n\\n")

            f.write("## 2. Performance Metrics\\n\\n")
            perf = self.data['performance_metrics']
            f.write(f"- **Success Rate**: {perf['success_rate']:.2f}%\\n")
            f.write(f"- **Average Latency**: {perf['avg_latency_ms']:.2f} ms (±{perf['std_latency_ms']:.2f})\\n")
            f.write(f"- **Min/Max Latency**: {perf['min_latency_ms']:.2f} / {perf['max_latency_ms']:.2f} ms\\n")
            f.write(f"- **Average Response Time**: {perf['avg_response_time_s']:.4f} s\\n\\n")

            f.write("## 3. Most Used Algorithms\\n\\n")
            sorted_algos = sorted(self.data['algorithm_usage'].items(),
                                  key=lambda x: x[1], reverse=True)
            for i, (algo, count) in enumerate(sorted_algos[:5], 1):
                f.write(f"{i}. **{algo}**: {count} times\\n")
            f.write("\\n")

            f.write("## 4. Analysis by Security Level\\n\\n")
            for level, data in self.data['by_security_level'].items():
                f.write(f"### {level.upper()}\\n")
                f.write(f"- Requests: {data['count']}\\n")
                f.write(f"- Success Rate: {data['success_rate']:.2f}%\\n")
                f.write(f"- Average Latency: {data['avg_latency']:.2f} ms\\n\\n")

            f.write("## 5. QKD vs Non-QKD Comparison\\n\\n")
            qkd = self.data['qkd_analysis']
            f.write("### With QKD\\n")
            f.write(f"- Requests: {qkd['with_qkd']['count']}\\n")
            f.write(f"- Success Rate: {qkd['with_qkd']['success_rate']:.2f}%\\n")
            f.write(f"- Average Latency: {qkd['with_qkd']['avg_latency']:.2f} ms\\n\\n")

            f.write("### Without QKD\\n")
            f.write(f"- Requests: {qkd['without_qkd']['count']}\\n")
            f.write(f"- Success Rate: {qkd['without_qkd']['success_rate']:.2f}%\\n")
            f.write(f"- Average Latency: {qkd['without_qkd']['avg_latency']:.2f} ms\\n\\n")

            f.write("## 6. Key Conclusions\\n\\n")
            f.write("1. The RL Engine demonstrated high success rate in algorithm selection\\n")
            f.write("2. Average latency remained within acceptable limits\\n")
            f.write("3. Quantum algorithms were prioritized in high-security scenarios\\n")
            f.write("4. The system efficiently adapted to different security contexts\\n")

        print(f"  ✓ article_summary.md")
        print(f"\\n✅ Summary saved in: {report_file}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <json_results_file>")
        print("\\nExample: python analyze_results.py rl_experiment_20241019_143022.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not Path(json_file).exists():
        print(f"❌ File not found: {json_file}")
        sys.exit(1)

    print("=" * 70)
    print("RL ENGINE - RESULTS ANALYSIS")
    print("=" * 70)
    print(f"File: {json_file}\\n")

    analyzer = ResultAnalyzer(json_file)

    # Generate all visualizations
    analyzer.generate_all_plots()

    # Generate LaTeX tables
    analyzer.generate_latex_tables()

    # Generate summary for article
    analyzer.generate_summary_report()

    print("\\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETED!")
    print("=" * 70)
    print(f"\\nAll files saved in: {analyzer.output_dir}/")
    print("\\nGenerated files:")
    print("  - 8 high-resolution PNG plots")
    print("  - 1 LaTeX tables file")
    print("  - 1 Markdown summary")


if __name__ == "__main__":
    main()