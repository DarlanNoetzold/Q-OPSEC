"""
Trust Result - Estruturas de dados para resultados de avaliação
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from core.trust_context import TrustContext


@dataclass
class SignalResult:
    """
    Resultado da avaliação de um signal individual
    """
    signal_name: str
    score: float  # [0, 1]
    confidence: float  # [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Valida valores"""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class TrustResult:
    """
    Resultado final da avaliação de confiança
    """
    trust_score: float  # [0, 1]
    trust_level: str  # VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
    confidence_interval: Tuple[float, float]  # (lower, upper)
    dimensions: Dict[str, float]  # {dimension: score}
    risk_flags: List[str]
    explainability: Dict[str, Any]
    trust_dna_value: str
    context: TrustContext
    signal_results: List[SignalResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict (para JSON response)"""
        return {
            "trust_score": round(self.trust_score, 4),
            "trust_level": self.trust_level,
            "confidence_interval": {
                "lower": round(self.confidence_interval[0], 4),
                "upper": round(self.confidence_interval[1], 4)
            },
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "risk_flags": self.risk_flags,
            "explainability": self.explainability,
            "trust_dna": self.trust_dna_value,
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "entity_id": self.context.entity_id,
                "source_id": self.context.source_id,
                "data_type": self.context.data_type,
                "environment": self.context.environment,
                "num_signals": len(self.signal_results)
            },
            "signals": [
                {
                    "name": sr.signal_name,
                    "score": round(sr.score, 4),
                    "confidence": round(sr.confidence, 4),
                    "metadata": sr.metadata
                }
                for sr in self.signal_results
            ]
        }

    def add_risk_flag(self, flag: str):
        """Adiciona risk flag"""
        if flag not in self.risk_flags:
            self.risk_flags.append(flag)

    def add_explanation(self, key: str, value: Any):
        """Adiciona explicação"""
        self.explainability[key] = value
