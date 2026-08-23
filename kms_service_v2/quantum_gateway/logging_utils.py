"""
Utilitários de logging estruturado e visualizações para protocolos QKD.

Gera logs em formato estruturado e visualizações PNG/PDF de:
- Distribuição de medições
- Gráficos de fidelidade
- Matriz de densidade
- Comparações de ruído
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Importações de visualização (opcionais)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class StructuredQKDLogger:
    """Logger estruturado para QKD com saída em arquivo e console."""
    
    def __init__(self, protocol_name: str, log_dir: str = "logs"):
        """
        Inicializa logger estruturado.
        
        Args:
            protocol_name: Nome do protocolo (BB84, E91, MDI-QKD)
            log_dir: Diretório para salvar logs
        """
        self.protocol_name = protocol_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{protocol_name}_{self.timestamp}.log"
        self.json_file = self.log_dir / f"{protocol_name}_{self.timestamp}.json"
        
        self.logger = logging.getLogger(f"qiskit_qkd.{protocol_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Handler para arquivo
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter estruturado
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.metadata = {
            "protocol": protocol_name,
            "timestamp": self.timestamp,
            "events": [],
            "metrics": {},
        }
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """
        Registra evento estruturado.
        
        Args:
            event_type: Tipo de evento (PREP, TRANS, MEAS, SIFT, ERROR_CORR, PRIV_AMP)
            message: Mensagem de log
            **kwargs: Dados adicionais (opcional)
        """
        level = kwargs.pop("level", "INFO")
        getattr(self.logger, level.lower())(message)
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            **kwargs
        }
        self.metadata["events"].append(event)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Registra métricas finais."""
        self.metadata["metrics"].update(metrics)
        self.logger.info(f"Métricas: {json.dumps(metrics, indent=2)}")
    
    def save_json_report(self):
        """Salva relatório em JSON."""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Relatório JSON salvo: {self.json_file}")
    
    def get_log_file(self) -> Path:
        """Retorna caminho do arquivo de log."""
        return self.log_file


class QKDVisualization:
    """Gerador de visualizações para análise de QKD."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Inicializa gerador de visualizações.
        
        Args:
            output_dir: Diretório para salvar gráficos
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib não está instalado. Execute: pip install matplotlib")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_measurement_histogram(self, measurements: np.ndarray, 
                                  title: str = "Distribuição de Medições",
                                  filename: str = "measurements.png"):
        """
        Plota histograma de distribuição de medições.
        
        Args:
            measurements: Array de medições (0s e 1s)
            title: Título do gráfico
            filename: Nome do arquivo a salvar
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        counts = [np.sum(measurements == 0), np.sum(measurements == 1)]
        bars = ax.bar(['0', '1'], counts, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Frequência', fontsize=12)
        ax.set_xlabel('Resultado de Medição', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_fidelity_over_time(self, fidelities: List[float],
                               title: str = "Fidelidade vs Número de Qubits",
                               filename: str = "fidelity.png"):
        """
        Plota gráfico de fidelidade.
        
        Args:
            fidelities: Lista de valores de fidelidade
            title: Título do gráfico
            filename: Nome do arquivo a salvar
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(fidelities))
        ax.plot(x, fidelities, 'o-', color='#2ecc71', linewidth=2, markersize=4)
        ax.axhline(y=0.95, color='red', linestyle='--', label='Limite de Segurança (0.95)')
        
        ax.set_ylabel('Fidelidade', fontsize=12)
        ax.set_xlabel('Número de Qubits', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0.8, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_qber_comparison(self, qber_with_noise: float, qber_without_noise: float,
                            threshold: float = 0.11,
                            filename: str = "qber_comparison.png"):
        """
        Plota comparação de QBER com e sem ruído.
        
        Args:
            qber_with_noise: QBER com ruído
            qber_without_noise: QBER sem ruído
            threshold: Limiar de segurança
            filename: Nome do arquivo a salvar
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Com Ruído', 'Sem Ruído']
        values = [qber_with_noise, qber_without_noise]
        colors = ['#e74c3c' if v > threshold else '#2ecc71' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', width=0.5)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Limiar ({threshold})')
        
        ax.set_ylabel('QBER (Quantum Bit Error Rate)', fontsize=12)
        ax.set_title('Comparação de QBER: Com vs Sem Ruído', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(values) * 1.2])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_noise_impact_summary(self, noise_stats: Dict[str, Any],
                                 filename: str = "noise_summary.png"):
        """
        Plota resumo do impacto de ruído.
        
        Args:
            noise_stats: Dicionário de estatísticas de ruído
            filename: Nome do arquivo a salvar
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Resumo de Impacto de Ruído', fontsize=14, fontweight='bold')
        
        # Gráfico 1: Contagem de erros
        ax = axes[0, 0]
        error_types = ['Depolarização', 'Damping', 'Dephasing']
        error_counts = [
            noise_stats.get('depolarization_errors', 0),
            noise_stats.get('amplitude_damping_events', 0),
            noise_stats.get('phase_damping_events', 0),
        ]
        ax.bar(error_types, error_counts, color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.7)
        ax.set_ylabel('Contagem', fontsize=11)
        ax.set_title('Erros por Tipo', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Falhas de detecção
        ax = axes[0, 1]
        detection_failures = noise_stats.get('detection_failures', 0)
        ax.text(0.5, 0.5, f'{detection_failures}', ha='center', va='center',
               fontsize=48, fontweight='bold', transform=ax.transAxes)
        ax.set_title('Falhas de Detecção', fontsize=12)
        ax.axis('off')
        
        # Gráfico 3: Fidelidade
        ax = axes[1, 0]
        fidelities = [
            noise_stats.get('initial_fidelity', 1.0),
            noise_stats.get('final_fidelity', 1.0),
        ]
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(['Inicial', 'Final'], fidelities, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Fidelidade', fontsize=11)
        ax.set_title('Fidelidade do Estado', fontsize=12)
        ax.set_ylim([0, 1.1])
        
        # Gráfico 4: Perda de Fidelidade
        ax = axes[1, 1]
        fidelity_loss = noise_stats.get('fidelity_loss_percent', 0.0)
        ax.text(0.5, 0.5, f'{fidelity_loss:.2f}%', ha='center', va='center',
               fontsize=48, fontweight='bold', transform=ax.transAxes)
        ax.set_title('Perda de Fidelidade', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_protocol_comparison(self, protocols_data: Dict[str, Dict[str, Any]],
                                filename: str = "protocol_comparison.png"):
        """
        Plota comparação entre protocolos.
        
        Args:
            protocols_data: Dict com dados dos protocolos {protocolo: {metricas}}
            filename: Nome do arquivo a salvar
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparação entre Protocolos QKD', fontsize=14, fontweight='bold')
        
        protocols = list(protocols_data.keys())
        
        # QBER
        ax = axes[0, 0]
        qbers = [protocols_data[p].get('qber', 0) for p in protocols]
        colors = ['#2ecc71' if q < 0.11 else '#e74c3c' for q in qbers]
        ax.bar(protocols, qbers, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0.11, color='red', linestyle='--', label='Limiar')
        ax.set_ylabel('QBER', fontsize=11)
        ax.set_title('Taxa de Erro Quântico', fontsize=12)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        # Comprimento de Chave
        ax = axes[0, 1]
        key_lengths = [protocols_data[p].get('key_length', 0) for p in protocols]
        ax.bar(protocols, key_lengths, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Bits', fontsize=11)
        ax.set_title('Comprimento Final de Chave', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Taxa de Transmissão
        ax = axes[1, 0]
        rates = [protocols_data[p].get('transmission_efficiency', 0) for p in protocols]
        ax.bar(protocols, rates, color='#f39c12', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Eficiência', fontsize=11)
        ax.set_title('Eficiência de Transmissão', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.tick_params(axis='x', rotation=45)
        
        # Sucesso
        ax = axes[1, 1]
        successes = [1 if protocols_data[p].get('successful', False) else 0 for p in protocols]
        colors_success = ['#2ecc71' if s else '#e74c3c' for s in successes]
        ax.bar(protocols, successes, color=colors_success, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Sucesso', fontsize=11)
        ax.set_title('Estado de Execução', fontsize=12)
        ax.set_ylim([0, 1.2])
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
