"""
Trust Configuration - Configurações centralizadas do Trust Engine
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SignalConfig:
    """
    Configuração de um signal individual
    """
    name: str
    enabled: bool = True
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustConfig:
    """
    Configuração global do Trust Engine V2
    """
    # Configurações de signals
    signals: Dict[str, SignalConfig] = field(default_factory=dict)

    # Thresholds para trust levels
    trust_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "very_low": 0.2,
        "low": 0.4,
        "medium": 0.6,
        "high": 0.8,
        "very_high": 1.0
    })

    # Configurações de storage
    max_history_per_entity: int = 100
    max_history_per_source: int = 100
    history_ttl_days: int = 30

    # Configurações de performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_concurrent_evaluations: int = 100

    # Configurações de aggregation
    scoring_strategy: str = "confidence_weighted"  # weighted_average, geometric_mean, etc.
    normalization_strategy: str = "sigmoid"  # clamp, min_max, z_score, etc.

    # Configurações de weighting
    enable_dynamic_weighting: bool = True
    enable_adaptive_weighting: bool = False

    def get_signal_config(self, signal_name: str) -> Optional[SignalConfig]:
        """
        Retorna configuração de um signal específico

        Args:
            signal_name: Nome do signal

        Returns:
            SignalConfig ou None se não encontrado
        """
        return self.signals.get(signal_name)

    def get_trust_level(self, trust_score: float) -> str:
        """
        Determina trust level baseado no score

        Args:
            trust_score: Score de confiança [0, 1]

        Returns:
            Trust level (VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
        """
        if trust_score < self.trust_thresholds["very_low"]:
            return "VERY_LOW"
        elif trust_score < self.trust_thresholds["low"]:
            return "LOW"
        elif trust_score < self.trust_thresholds["medium"]:
            return "MEDIUM"
        elif trust_score < self.trust_thresholds["high"]:
            return "HIGH"
        else:
            return "VERY_HIGH"


def create_default_config() -> TrustConfig:
    """
    Cria configuração padrão do Trust Engine

    Returns:
        TrustConfig com valores padrão
    """
    config = TrustConfig()

    # Configuração de signals
    config.signals = {
        "temporal": SignalConfig(
            name="temporal",
            enabled=True,
            weight=1.2,
            params={
                "decay_lambda": 0.0001,
                "half_life_hours": 168.0  # 7 dias
            }
        ),
        "temporal_drift": SignalConfig(
            name="temporal_drift",
            enabled=True,
            weight=1.5,
            params={}
        ),
        "source_reliability": SignalConfig(
            name="source_reliability",
            enabled=True,
            weight=1.3,
            params={
                "min_history_for_confidence": 10
            }
        ),
        "source_consistency": SignalConfig(
            name="source_consistency",
            enabled=True,
            weight=1.0,
            params={
                "consistency_threshold": 0.7
            }
        ),
        "semantic_consistency": SignalConfig(
            name="semantic_consistency",
            enabled=True,
            weight=1.1,
            params={}
        ),
        "semantic_drift": SignalConfig(
            name="semantic_drift",
            enabled=True,
            weight=0.9,
            params={
                "drift_threshold": 0.5
            }
        ),
        "anomaly_detection": SignalConfig(
            name="anomaly_detection",
            enabled=True,
            weight=1.4,
            params={
                "entropy_threshold": 4.5,
                "max_payload_size": 10000,
                "min_payload_size": 10
            }
        ),
        "consistency": SignalConfig(
            name="consistency",
            enabled=True,
            weight=1.0,
            params={}
        ),
        "context_alignment": SignalConfig(
            name="context_alignment",
            enabled=True,
            weight=1.1,
            params={
                "valid_environments": ["production", "staging", "development", "test"],
                "valid_data_types": ["event", "profile", "transaction", "log", "metric"]
            }
        ),
        "context_stability": SignalConfig(
            name="context_stability",
            enabled=True,
            weight=0.8,
            params={
                "min_history_for_stability": 3
            }
        )
    }

    return config
