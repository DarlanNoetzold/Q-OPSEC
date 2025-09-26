# app/models/database.py
from beanie import Document, Indexed
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class ModelStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"


class PredictionRecord(Document):
    """Documento para armazenar registros de predições"""

    request_id: Indexed(str) = Field(..., description="ID único da requisição")
    ml_name: Indexed(str) = Field(..., description="Nome do modelo usado")
    ml_version: str = Field(..., description="Versão do modelo")
    input_data: Dict[str, Any] = Field(..., description="Dados de entrada")
    prediction: Optional[Dict[str, Any]] = Field(None, description="Resultado da predição")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Scores de confiança")
    status: PredictionStatus = Field(default=PredictionStatus.PENDING)
    error_message: Optional[str] = Field(None, description="Mensagem de erro se houver")
    processing_time_ms: Optional[float] = Field(None, description="Tempo de processamento em ms")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "predictions"
        indexes = [
            [("request_id", 1)],
            [("ml_name", 1), ("created_at", -1)],
            [("status", 1), ("created_at", -1)],
            [("created_at", -1)],
        ]


class ModelRecord(Document):
    """Documento para armazenar informações dos modelos"""

    name: Indexed(str) = Field(..., description="Nome do modelo")
    version: str = Field(..., description="Versão do modelo")
    algorithm: str = Field(..., description="Algoritmo usado")
    file_path: str = Field(..., description="Caminho do arquivo do modelo")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados do modelo")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Métricas de performance")
    status: ModelStatus = Field(default=ModelStatus.ACTIVE)
    is_default: bool = Field(default=False, description="Se é o modelo padrão")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = Field(None, description="Última vez que foi usado")
    usage_count: int = Field(default=0, description="Número de vezes usado")

    class Settings:
        name = "models"
        indexes = [
            [("name", 1), ("version", -1)],
            [("status", 1)],
            [("is_default", 1)],
            [("created_at", -1)],
        ]


class MetricRecord(Document):
    """Documento para armazenar métricas da API"""

    endpoint: Indexed(str) = Field(..., description="Endpoint da API")
    method: str = Field(..., description="Método HTTP")
    status_code: int = Field(..., description="Código de status HTTP")
    response_time_ms: float = Field(..., description="Tempo de resposta em ms")
    ml_name: Optional[str] = Field(None, description="Nome do modelo usado")
    user_agent: Optional[str] = Field(None, description="User agent da requisição")
    ip_address: Optional[str] = Field(None, description="IP do cliente")
    error_message: Optional[str] = Field(None, description="Mensagem de erro se houver")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "metrics"
        indexes = [
            [("endpoint", 1), ("timestamp", -1)],
            [("status_code", 1), ("timestamp", -1)],
            [("timestamp", -1)],
        ]


# Modelos Pydantic usados pelos handlers de erro/validação
class ErrorResponse(BaseModel):
    """Modelo para resposta de erro padrão dos handlers"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_config = {"protected_namespaces": ()}


class ValidationErrorDetail(BaseModel):
    """Detalhe de erro de validação por campo"""
    field: str
    message: str
    value: Optional[Any] = None
    model_config = {"protected_namespaces": ()}


class ValidationErrorResponse(BaseModel):
    """Resposta de erro de validação (HTTP 422)"""
    error: str = "validation_error"
    details: List[ValidationErrorDetail]
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_config = {"protected_namespaces": ()}