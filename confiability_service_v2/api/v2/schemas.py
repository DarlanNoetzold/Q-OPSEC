"""
Schemas - Contratos de entrada/saída da API V2 (Pydantic)
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class TrustRequest(BaseModel):
    """Schema de requisição para avaliação de confiança"""

    payload: Dict[str, Any] = Field(
        ...,
        description="Dados da informação a ser avaliada",
        example={
            "claim": "A Terra é plana",
            "author": "user_123",
            "content": "Evidências científicas comprovam..."
        }
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="Metadados contextuais",
        example={
            "source_id": "src_news_portal",
            "entity_id": "ent_flat_earth_claim",
            "timestamp": "2026-01-30T10:00:00Z",
            "data_type": "claim",
            "environment": "production"
        }
    )

    history: Optional[Dict[str, Any]] = Field(
        default={},
        description="Histórico ou contexto adicional",
        example={
            "previous_evaluations": 5,
            "last_trust_score": 0.3
        }
    )

    @validator('payload')
    def validate_payload(cls, v):
        if not v:
            raise ValueError("Payload não pode estar vazio")
        return v

    class Config:
        schema_extra = {
            "example": {
                "payload": {
                    "claim": "Vacinas causam autismo",
                    "source": "blog_xyz"
                },
                "metadata": {
                    "source_id": "src_blog_xyz",
                    "entity_id": "ent_vaccine_claim_001",
                    "timestamp": "2026-01-30T10:00:00Z",
                    "data_type": "health_claim"
                }
            }
        }


class TrustResponse(BaseModel):
    """Schema de resposta da avaliação de confiança"""

    trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiança (0-1)"
    )

    trust_level: str = Field(
        ...,
        description="Nível de confiança categórico",
        example="MEDIUM"
    )

    confidence_interval: tuple[float, float] = Field(
        ...,
        description="Intervalo de confiança [min, max]",
        example=[0.45, 0.65]
    )

    risk_flags: List[str] = Field(
        default=[],
        description="Flags de risco identificados",
        example=["high_entropy", "source_inconsistency"]
    )

    trust_dna: str = Field(
        ...,
        description="Fingerprint único da avaliação (Trust DNA)",
        example="a3f5c9d2e1b4"
    )

    dimensions: Dict[str, float] = Field(
        ...,
        description="Scores por dimensão de confiança",
        example={
            "temporal": 0.85,
            "source": 0.60,
            "semantic": 0.45,
            "consistency": 0.70
        }
    )

    explainability: Dict[str, Any] = Field(
        ...,
        description="Explicações detalhadas da avaliação",
        example={
            "summary": "Confiança MÉDIA devido a inconsistências semânticas",
            "signals": {
                "temporal_signal": {
                    "score": 0.85,
                    "reason": "Informação recente (2h)"
                }
            }
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "trust_score": 0.55,
                "trust_level": "MEDIUM",
                "confidence_interval": [0.45, 0.65],
                "risk_flags": ["semantic_drift", "low_source_reliability"],
                "trust_dna": "a3f5c9d2e1b4",
                "dimensions": {
                    "temporal": 0.85,
                    "source": 0.40,
                    "semantic": 0.50,
                    "consistency": 0.65
                },
                "explainability": {
                    "summary": "Confiança MÉDIA. Source com histórico inconsistente.",
                    "top_signals": ["temporal_signal", "source_reliability"],
                    "signals": {}
                }
            }
        }


class HealthResponse(BaseModel):
    """Schema de resposta do health check"""

    status: str = Field(..., example="healthy")
    version: str = Field(..., example="2.0.0")
    signals_count: int = Field(..., example=10)
    signals: List[str] = Field(
        ...,
        example=["temporal_signal", "source_reliability", "semantic_consistency"]
    )


class ConfigResponse(BaseModel):
    """Schema de resposta da configuração"""

    trust_levels: Dict[str, tuple[float, float]] = Field(
        ...,
        description="Thresholds dos níveis de confiança"
    )

    signals: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Configuração de cada sinal"
    )

    storage: Dict[str, int] = Field(
        ...,
        description="Configurações de armazenamento"
    )


class TrustHistoryRequest(BaseModel):
    """Schema para consulta de histórico"""

    key: str = Field(
        ...,
        description="ID da entidade ou source",
        example="ent_vaccine_claim_001"
    )

    key_type: str = Field(
        ...,
        description="Tipo da chave: 'entity' ou 'source'",
        example="entity"
    )

    limit: Optional[int] = Field(
        default=50,
        ge=1,
        le=1000,
        description="Número máximo de eventos a retornar"
    )

    @validator('key_type')
    def validate_key_type(cls, v):
        if v not in ["entity", "source"]:
            raise ValueError("key_type deve ser 'entity' ou 'source'")
        return v


class TrustHistoryResponse(BaseModel):
    """Schema de resposta do histórico"""

    key: str
    key_type: str
    events: List[Dict[str, Any]]
    statistics: Dict[str, float] = Field(
        ...,
        example={
            "avg_trust_score": 0.65,
            "min_trust_score": 0.30,
            "max_trust_score": 0.85,
            "trust_volatility": 0.15,
            "total_evaluations": 42
        }
    )