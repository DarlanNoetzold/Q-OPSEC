"""
Trust Engine V2 - Schemas de Request/Response
Validação de entrada usando Pydantic
"""
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Optional


class TrustEvaluationRequest(BaseModel):
    """
    Schema para requisição de avaliação de confiança
    """
    payload: Dict[str, Any] = Field(
        ...,
        description="Dados a serem avaliados"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados contextuais (source_id, entity_id, timestamp, etc.)"
    )

    history: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Histórico opcional para contexto adicional"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "payload": {
                    "claim": "User reported suspicious activity",
                    "details": {"ip": "192.168.1.1", "action": "login"}
                },
                "metadata": {
                    "source_id": "security_system_1",
                    "entity_id": "user_12345",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "data_type": "security_event",
                    "environment": "production"
                }
            }
        }


def validate_request(model_class: type[BaseModel], data: dict) -> tuple[BaseModel | None, ValidationError | None]:
    """
    Valida dados de entrada contra um modelo Pydantic

    Returns:
        (model_instance, None) se válido
        (None, validation_error) se inválido
    """
    try:
        instance = model_class(**data)
        return instance, None
    except ValidationError as e:
        return None, e