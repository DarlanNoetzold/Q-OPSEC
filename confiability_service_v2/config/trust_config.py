"""
Centralized configuration for Trust Engine V2.
"""
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class SignalConfig:
    """Configuration for a trust signal."""
    enabled: bool = True
    weight: float = 1.0
    params: Dict = field(default_factory=dict)


@dataclass
class TrustConfig:
    """Main trust engine configuration."""

    # Signal configurations
    signals: Dict[str, SignalConfig] = field(default_factory=lambda: {
        "temporal": SignalConfig(
            enabled=True,
            weight=1.2,
            params={
                "decay_lambda": 0.0001,  # decay rate per second
                "max_age_hours": 720  # 30 days
            }
        ),
        "semantic": SignalConfig(
            enabled=True,
            weight=1.5,
            params={
                "similarity_threshold": 0.7,
                "max_history": 10
            }
        ),
        "source": SignalConfig(
            enabled=True,
            weight=1.3,
            params={
                "min_history": 3,
                "reputation_decay": 0.95
            }
        ),
        "consistency": SignalConfig(
            enabled=True,
            weight=1.4,
            params={
                "lookback_window": 20,
                "contradiction_penalty": 0.3
            }
        ),
        "anomaly": SignalConfig(
            enabled=True,
            weight=1.1,
            params={
                "z_score_threshold": 2.5,
                "min_samples": 10
            }
        ),
        "context": SignalConfig(
            enabled=True,
            weight=1.0,
            params={
                "expected_contexts": ["prod", "staging", "dev"]
            }
        )
    })

    # Trust level thresholds
    trust_levels: Dict[str, tuple] = field(default_factory=lambda: {
        "CRITICAL": (0.0, 0.3),
        "LOW": (0.3, 0.5),
        "MEDIUM": (0.5, 0.75),
        "HIGH": (0.75, 0.9),
        "VERIFIED": (0.9, 1.0)
    })

    # Storage settings
    storage: Dict = field(default_factory=lambda: {
        "max_history_per_entity": 100,
        "retention_days": 90,
        "enable_graph": True
    })

    # Performance settings
    performance: Dict = field(default_factory=lambda: {
        "deep_analysis_threshold": 0.6,  # only deep dive if initial score < this
        "max_signal_timeout_ms": 500,
        "enable_caching": True,
        "cache_ttl_seconds": 300
    })

    def get_signal_config(self, signal_name: str) -> SignalConfig:
        """Get configuration for a specific signal."""
        return self.signals.get(signal_name, SignalConfig(enabled=False))

    def get_trust_level(self, score: float) -> str:
        """Determine trust level from score."""
        for level, (min_score, max_score) in self.trust_levels.items():
            if min_score <= score < max_score:
                return level
        return "UNKNOWN"

    def enabled_signals(self) -> List[str]:
        """Get list of enabled signal names."""
        return [name for name, config in self.signals.items() if config.enabled]


# Global config instance
default_config = TrustConfig()