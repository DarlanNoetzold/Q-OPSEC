"""
Models - Modelos de domínio para a API V2
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class TrustEvaluationModel:
    """Modelo de domínio para uma avaliação de confiança"""

    trust_score: float
    trust_level: str
    confidence_interval: tuple[float, float]
    risk_flags: List[str]
    trust_dna: str
    dimensions: Dict[str, float]
    explainability: Dict[str, Any]

    # Metadata
    evaluated_at: datetime
    entity_id: Optional[str] = None
    source_id: Optional[str] = None
    payload_hash: Optional[str] = None

    def to_dict(self) -> dict:
        """Converte para dicionário"""
        return {
            "trust_score": self.trust_score,
            "trust_level": self.trust_level,
            "confidence_interval": self.confidence_interval,
            "risk_flags": self.risk_flags,
            "trust_dna": self.trust_dna,
            "dimensions": self.dimensions,
            "explainability": self.explainability,
            "evaluated_at": self.evaluated_at.isoformat(),
            "entity_id": self.entity_id,
            "source_id": self.source_id,
            "payload_hash": self.payload_hash
        }

    @property
    def is_trustworthy(self) -> bool:
        """Verifica se a informação é confiável (HIGH ou VERIFIED)"""
        return self.trust_level in ["HIGH", "VERIFIED"]

    @property
    def has_critical_risks(self) -> bool:
        """Verifica se há flags de risco crítico"""
        critical_keywords = ["critical", "severe", "high_risk", "anomaly"]
        return any(
            keyword in flag.lower()
            for flag in self.risk_flags
            for keyword in critical_keywords
        )


@dataclass
class SignalEvaluationModel:
    """Modelo para resultado de um sinal individual"""

    name: str
    score: float
    confidence: float
    weight: float
    contribution: float
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "confidence": self.confidence,
            "weight": self.weight,
            "contribution": self.contribution,
            "metadata": self.metadata
        }


@dataclass
class TrustHistoryModel:
    """Modelo para histórico de confiança de uma entidade/source"""

    key: str  # entity_id ou source_id
    key_type: str  # "entity" ou "source"
    events: List[Dict[str, Any]]

    # Estatísticas agregadas
    avg_trust_score: float
    min_trust_score: float
    max_trust_score: float
    trust_volatility: float  # desvio padrão
    total_evaluations: int

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "key_type": self.key_type,
            "events": self.events,
            "statistics": {
                "avg_trust_score": self.avg_trust_score,
                "min_trust_score": self.min_trust_score,
                "max_trust_score": self.max_trust_score,
                "trust_volatility": self.trust_volatility,
                "total_evaluations": self.total_evaluations
            }
        }


@dataclass
class TrustDimensionModel:
    """Modelo para uma dimensão de confiança"""

    name: str
    score: float
    weight: float
    description: str
    contributing_signals: List[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "description": self.description,
            "contributing_signals": self.contributing_signals
        }