"""
API endpoints for the Classification Agent.
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import requests
from pathlib import Path
import json

from ...core.config import settings
from ...core.security import get_current_user, require_auth
from ...core.logging import get_logger
from ...services.model_service import model_service, ModelLoadError, PredictionError
from ...services.metrics_service import metrics_service
from ...utils.exceptions import (
    ModelNotLoadedException, PredictionException, ValidationException
)

logger = get_logger(__name__)
router = APIRouter()

# Metrics directory configuration
METRICS_ROOT = Path(r"C:\Projetos\Q-OPSEC\classify_scheduler\models\metrics")
IMAGE_ALLOWED = {
    "all_models_accuracy.png",
    "all_models_f1score.png",
    "accuracy_vs_f1.png",
    "best_model_confusion_matrix.png",
    "top10_models_ranking.png",
}


# ============================================================================
# Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Resposta do health check"""
    status: str = Field(..., description="Status do serviço (ok, degraded, error)")
    version: str = Field(..., description="Versão da API")
    model_loaded: bool = Field(..., description="Se há um modelo carregado")
    model_name: Optional[str] = Field(None, description="Nome do modelo carregado")
    uptime_seconds: float = Field(..., description="Tempo de uptime em segundos")
    model_config = {"protected_namespaces": ()}


class ModelReloadRequest(BaseModel):
    """Request para reload do modelo"""
    force: bool = Field(False, description="Forçar reload mesmo se o modelo já estiver atualizado")
    model_config = {"protected_namespaces": ()}


class ModelReloadResponse(BaseModel):
    """Resposta do reload do modelo"""
    status: str = Field(..., description="Status da operação (success, error)")
    model_name: Optional[str] = Field(None, description="Nome do modelo carregado")
    previous_model: Optional[str] = Field(None, description="Nome do modelo anterior")
    message: str = Field(..., description="Mensagem descritiva")
    model_config = {"protected_namespaces": ()}


class PredictionRequest(BaseModel):
    """Request de predição"""
    data: Any = Field(..., description="Objeto ou lista de objetos com os dados para predição")
    return_probabilities: bool = Field(True, description="Se deve retornar probabilidades por classe")
    send_to_rl: bool = Field(False, description="Se deve enviar resultado para RL Engine")
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [
                {
                    "data": {
                        "feature1": 0.5,
                        "feature2": "value",
                        "feature3": 100
                    },
                    "return_probabilities": True,
                    "send_to_rl": False
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    """Resultado individual de predição"""
    label: str = Field(..., description="Classe predita")
    confidence: Optional[float] = Field(None, description="Confiança da predição (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilidades por classe")
    input_hash: Optional[str] = Field(None, description="Hash do input para rastreamento")
    rl_decision: Optional[str] = Field(None, description="Decisão do RL Engine (se send_to_rl=true)")
    model_config = {"protected_namespaces": ()}


class PredictResponse(BaseModel):
    """Resposta completa de predição"""
    results: List[PredictionResult] = Field(..., description="Lista de resultados de predição")
    model_name: Optional[str] = Field(None, description="Nome do modelo usado")
    model_version: Optional[str] = Field(None, description="Versão do modelo")
    classes: List[str] = Field([], description="Lista de classes possíveis")
    prediction_time_ms: float = Field(..., description="Tempo de predição em milissegundos")
    batch_size: int = Field(..., description="Tamanho do batch processado")
    model_config = {"protected_namespaces": ()}


class MetricsResponse(BaseModel):
    """Métricas do serviço"""
    total_requests: int = Field(..., description="Total de requisições recebidas")
    total_predictions: int = Field(..., description="Total de predições realizadas")
    average_response_time_ms: float = Field(..., description="Tempo médio de resposta em ms")
    error_rate: float = Field(..., description="Taxa de erro (0-1)")
    model_reload_count: int = Field(..., description="Número de reloads do modelo")
    uptime_seconds: float = Field(..., description="Tempo de uptime em segundos")
    last_prediction_at: Optional[datetime] = Field(None, description="Timestamp da última predição")
    current_model: Optional[str] = Field(None, description="Nome do modelo atual")
    model_config = {"protected_namespaces": ()}


# ============================================================================
# Helper Functions for Training Metrics
# ============================================================================

def _list_training_sessions() -> List[str]:
    """Lista todas as sessões de treinamento disponíveis"""
    if not METRICS_ROOT.exists():
        return []
    sessions = []
    for p in METRICS_ROOT.iterdir():
        if p.is_dir() and (p / "training_summary.json").exists():
            sessions.append(p.name)
    sessions.sort(reverse=True)  # Mais recentes primeiro
    return sessions


def _get_latest_training_session() -> Optional[Path]:
    """Retorna o path da sessão de treinamento mais recente"""
    sessions = _list_training_sessions()
    if not sessions:
        return None
    return METRICS_ROOT / sessions[0]


# ============================================================================
# Health & Model Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check do serviço",
    description="Verifica o status de saúde do serviço, incluindo se o modelo está carregado e tempo de uptime.",
    responses={
        200: {
            "description": "Serviço saudável",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "version": "1.0.0",
                        "model_loaded": True,
                        "model_name": "random_forest_v1.2.3",
                        "uptime_seconds": 3600.5
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Verifica o status de saúde do serviço.

    Retorna informações sobre:
    - Status geral do serviço
    - Versão da API
    - Se há modelo carregado
    - Nome do modelo atual
    - Tempo de uptime
    """
    health_data = metrics_service.get_health_status()
    return HealthResponse(
        status=health_data.get("status", "unknown"),
        version=settings.api_version,
        model_loaded=health_data.get("model_loaded", model_service.is_model_loaded()),
        model_name=health_data.get("model_name", model_service.model_name),
        uptime_seconds=health_data.get("uptime_seconds", 0.0),
    )


@router.get(
    "/model",
    tags=["Model"],
    summary="Informações do modelo carregado",
    description="Retorna informações detalhadas sobre o modelo atualmente carregado, incluindo nome, versão, classes e metadados.",
    responses={
        200: {
            "description": "Informações do modelo",
            "content": {
                "application/json": {
                    "example": {
                        "model_name": "random_forest_v1.2.3",
                        "model_version": "1.2.3",
                        "classes": ["class_a", "class_b", "class_c"],
                        "required_columns": ["feature1", "feature2", "feature3"],
                        "loaded_at": "2024-01-15T10:30:00",
                        "meta": {"accuracy": 0.95, "f1_score": 0.93}
                    }
                }
            }
        },
        503: {"description": "Nenhum modelo carregado"}
    }
)
async def get_model_info(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retorna informações detalhadas do modelo carregado.

    Inclui:
    - Nome e versão do modelo
    - Classes disponíveis
    - Colunas/features requeridas
    - Timestamp de carregamento
    - Metadados adicionais (métricas, etc.)
    """
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    info = model_service.get_model_info()
    if not info:
        raise ModelNotLoadedException("Model information not available")

    return info


@router.post(
    "/model/reload",
    response_model=ModelReloadResponse,
    tags=["Model"],
    summary="Recarrega o modelo mais recente",
    description="""
Recarrega o modelo mais recente disponível.

- Se `force=false` (padrão): só recarrega se houver uma versão mais nova
- Se `force=true`: força o reload mesmo se o modelo já estiver atualizado

Requer autenticação.
""",
    responses={
        200: {
            "description": "Modelo recarregado com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "model_name": "random_forest_v1.2.4",
                        "previous_model": "random_forest_v1.2.3",
                        "message": "Model reloaded successfully"
                    }
                }
            }
        },
        401: {"description": "Não autorizado"},
        500: {"description": "Falha ao carregar modelo"}
    }
)
async def reload_model(
    request: ModelReloadRequest = ModelReloadRequest(),
    user: Dict[str, Any] = Depends(require_auth),
):
    """
    Recarrega o modelo mais recente.

    Args:
        request: Configuração do reload (force=true para forçar)
        user: Usuário autenticado

    Returns:
        Status do reload e informações do modelo
    """
    try:
        previous_model = model_service.model_name if model_service.is_model_loaded() else None
        reloaded = await model_service.load_latest_model(force=request.force)

        if reloaded:
            metrics_service.record_model_reload(model_service.model_name)
            message = "Model reloaded successfully"
            logger.info("Model reloaded", new_model=model_service.model_name, previous_model=previous_model)
        else:
            message = "Model was already up to date"

        return ModelReloadResponse(
            status="success",
            model_name=model_service.model_name,
            previous_model=previous_model,
            message=message,
        )

    except ModelLoadError as e:
        logger.error("Model reload failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        )


@router.get(
    "/model/manifest",
    tags=["Model"],
    summary="Manifest do modelo (schema de entrada/saída)",
    description="""
Retorna o manifest completo do modelo, incluindo:

- Schema de entrada (features requeridas e tipos)
- Schema de saída (formato da predição)
- Classes disponíveis
- Metadados do modelo
""",
    responses={
        200: {
            "description": "Manifest do modelo",
            "content": {
                "application/json": {
                    "example": {
                        "model_name": "random_forest_v1.2.3",
                        "classes": ["class_a", "class_b"],
                        "required_columns": ["feature1", "feature2"],
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "feature1": {"type": "any"},
                                "feature2": {"type": "any"}
                            },
                            "required": ["feature1", "feature2"]
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "enum": ["class_a", "class_b"]},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "probabilities": {"type": "object"}
                            }
                        }
                    }
                }
            }
        },
        503: {"description": "Nenhum modelo carregado"}
    }
)
async def get_model_manifest(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retorna o manifest do modelo com schemas de entrada e saída.

    Útil para:
    - Validação de inputs
    - Geração de documentação
    - Integração com outros sistemas
    """
    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    info = model_service.get_model_info()
    if not info:
        raise ModelNotLoadedException("Model information not available")

    classes = info.get("classes", [])
    required_columns = info.get("required_columns", [])
    manifest = {
        "model_name": info.get("saved_model_name") or info.get("model_name") or model_service.model_name,
        "classes": classes,
        "required_columns": required_columns,
        "input_schema": {
            "type": "object",
            "properties": {col: {"type": "any"} for col in required_columns},
            "required": required_columns,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": classes},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "probabilities": {
                    "type": "object",
                    "properties": {cls: {"type": "number"} for cls in classes},
                },
                "rl_decision": {"type": ["string", "null"]},
            },
        },
        "metadata": info.get("meta", {}),
        "loaded_at": info.get("loaded_at"),
    }
    return manifest


# ============================================================================
# Prediction Endpoint
# ============================================================================

@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Executa predição/classificação",
    description="""
Executa predição usando o modelo carregado.

### Fluxo
1. **Validação**: Valida o input contra o schema do modelo
2. **Predição**: Executa a predição e retorna label + probabilidades
3. **RL Engine** (opcional): Se `send_to_rl=true`, envia resultado para RL Engine

### Input
- `data`: Objeto ou lista de objetos com as features
- `return_probabilities`: Se deve retornar probabilidades (padrão: true)
- `send_to_rl`: Se deve enviar para RL Engine (padrão: false)

### Output
- `results`: Lista de predições (label, confidence, probabilities)
- `model_name`, `model_version`: Informações do modelo usado
- `prediction_time_ms`: Tempo de processamento
""",
    responses={
        200: {
            "description": "Predição realizada com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "label": "class_a",
                                "confidence": 0.85,
                                "probabilities": {"class_a": 0.85, "class_b": 0.15},
                                "input_hash": "abc123",
                                "rl_decision": None
                            }
                        ],
                        "model_name": "random_forest_v1.2.3",
                        "model_version": "1.2.3",
                        "classes": ["class_a", "class_b"],
                        "prediction_time_ms": 12.5,
                        "batch_size": 1
                    }
                }
            }
        },
        400: {"description": "Erro de validação do input"},
        401: {"description": "Não autorizado"},
        503: {"description": "Modelo não carregado"},
        500: {"description": "Erro interno durante predição"}
    }
)
async def predict(
    request: PredictionRequest,
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Executa predição com o modelo carregado.

    Args:
        request: Dados de entrada e configurações
        http_request: Request HTTP (para logging)
        user: Usuário autenticado

    Returns:
        Resultados da predição com labels, probabilidades e metadados
    """
    start_time = time.time()

    if not model_service.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded")

    try:
        validation_errors = model_service.validate_input(request.data)
        if validation_errors:
            raise ValidationException(
                "Input validation failed",
                details={"validation_errors": validation_errors},
            )

        labels, probabilities, input_hashes = model_service.predict(
            request.data, include_probabilities=request.return_probabilities
        )

        batch_size = 1 if isinstance(request.data, dict) else len(request.data)
        metrics_service.record_prediction(model_service.model_name, batch_size)

        rl_decisions: List[Optional[str]] = [None] * batch_size

        if request.send_to_rl:
            RL_ENGINE_URL = getattr(settings, "rl_engine_url", "http://localhost:9009/act")

            rl_payload = {
                "request_id": request.data.get("request_id_resolved") if isinstance(request.data, dict) else None,
                "source": "node-A",
                "destination": "http://localhost:9000",
                "security_level": labels[0],
                "risk_score": request.data.get("risk_score", 0.0) if isinstance(request.data, dict) else 0.0,
                "conf_score": request.data.get("conf_score", 0.0) if isinstance(request.data, dict) else 0.0,
                "dst_props": {
                    "hardware": ["QKD"],
                    "compliance": ["GDPR"],
                    "max_latency_ms": 10
                }
            }

            try:
                rl_response = requests.post(RL_ENGINE_URL, json=rl_payload, timeout=5.0)
                rl_response.raise_for_status()
                rl_result = rl_response.json()
                rl_decisions[0] = rl_result.get("action", "unknown")
                logger.info("RL Engine response received", result=rl_result)
            except Exception as e:
                logger.warning("RL Engine unavailable, fallback ignored", error=str(e))

        results: List[PredictionResult] = []
        for i, label in enumerate(labels):
            probs = probabilities[i] if probabilities and i < len(probabilities) else None
            conf = probs.get(label) if probs else None
            results.append(
                PredictionResult(
                    label=str(label),
                    confidence=conf,
                    probabilities=probs,
                    input_hash=input_hashes[i] if i < len(input_hashes) else None,
                    rl_decision=rl_decisions[i],
                )
            )

        prediction_time = (time.time() - start_time) * 1000.0

        resp = PredictResponse(
            results=results,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            classes=model_service.classes or [],
            prediction_time_ms=round(prediction_time, 2),
            batch_size=batch_size,
        )

        logger.info(
            "Prediction completed",
            request_id=getattr(http_request.state, "request_id", None),
            model_name=model_service.model_name,
            batch_size=batch_size,
            prediction_time_ms=resp.prediction_time_ms,
        )
        return resp

    except PredictionError as e:
        logger.error("Prediction failed", error=str(e))
        raise PredictionException(str(e))
    except Exception as e:
        logger.error("Unexpected error during prediction", error=str(e), exc_info=True)
        raise PredictionException(f"Prediction failed: {str(e)}")


# ============================================================================
# Monitoring Endpoints
# ============================================================================

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
    summary="Métricas do serviço",
    description="""
Retorna métricas agregadas do serviço, incluindo:

- Total de requisições e predições
- Tempo médio de resposta
- Taxa de erro
- Número de reloads do modelo
- Uptime
- Timestamp da última predição
""",
    responses={
        200: {
            "description": "Métricas do serviço",
            "content": {
                "application/json": {
                    "example": {
                        "total_requests": 1500,
                        "total_predictions": 1450,
                        "average_response_time_ms": 15.3,
                        "error_rate": 0.02,
                        "model_reload_count": 3,
                        "uptime_seconds": 86400.0,
                        "last_prediction_at": "2024-01-15T14:30:00",
                        "current_model": "random_forest_v1.2.3"
                    }
                }
            }
        },
        401: {"description": "Não autorizado"}
    }
)
async def get_metrics(user: Dict[str, Any] = Depends(require_auth)):
    """
    Retorna métricas agregadas do serviço.

    Útil para:
    - Monitoramento de performance
    - Alertas e dashboards
    - Análise de uso
    """
    m = metrics_service.get_metrics()
    return MetricsResponse(
        total_requests=m.get("total_requests", 0),
        total_predictions=m.get("total_predictions", 0),
        average_response_time_ms=m.get("average_response_time_ms", 0.0),
        error_rate=m.get("error_rate", 0.0),
        model_reload_count=m.get("model_reload_count", 0),
        uptime_seconds=m.get("uptime_seconds", 0.0),
        last_prediction_at=m.get("last_prediction_at"),
        current_model=m.get("current_model"),
    )


# ============================================================================
# Training Metrics Endpoints
# ============================================================================

@router.get(
    "/training/sessions",
    tags=["Training Metrics"],
    summary="Lista sessões de treinamento",
    description="Retorna lista de todas as sessões de treinamento disponíveis (diretórios com training_summary.json).",
    responses={
        200: {
            "description": "Lista de sessões",
            "content": {
                "application/json": {
                    "example": {
                        "sessions": ["20240115_143000", "20240114_120000"],
                        "total": 2
                    }
                }
            }
        }
    }
)
async def list_training_sessions(user: Dict[str, Any] = Depends(get_current_user)):
    """Lista todas as sessões de treinamento disponíveis."""
    sessions = _list_training_sessions()
    return JSONResponse(content={"sessions": sessions, "total": len(sessions)})


@router.get(
    "/training/latest",
    tags=["Training Metrics"],
    summary="Summary da sessão de treinamento mais recente",
    description="Retorna o arquivo training_summary.json da sessão mais recente.",
    responses={
        200: {"description": "Summary do treinamento"},
        404: {"description": "Nenhuma sessão de treinamento disponível"}
    }
)
async def get_latest_training_summary(user: Dict[str, Any] = Depends(get_current_user)):
    """Retorna o summary da sessão de treinamento mais recente."""
    latest = _get_latest_training_session()
    if latest is None:
        raise HTTPException(status_code=404, detail="No training metrics available")

    summary_path = latest / "training_summary.json"
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error("Failed to read training summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to read training summary: {str(e)}")


@router.get(
    "/training/{session_id}/summary",
    tags=["Training Metrics"],
    summary="Summary de uma sessão específica",
    description="Retorna o training_summary.json de uma sessão específica pelo ID.",
    responses={
        200: {"description": "Summary do treinamento"},
        404: {"description": "Sessão não encontrada"}
    }
)
async def get_training_session_summary(
    session_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Retorna o summary de uma sessão de treinamento específica."""
    session_path = METRICS_ROOT / session_id
    summary_path = session_path / "training_summary.json"

    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Session not found or summary missing")

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error("Failed to read training summary", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to read training summary: {str(e)}")


@router.get(
    "/training/images",
    tags=["Training Metrics"],
    summary="Lista imagens disponíveis",
    description="Retorna lista de imagens (gráficos) disponíveis para download.",
    responses={
        200: {
            "description": "Lista de imagens disponíveis",
            "content": {
                "application/json": {
                    "example": {
                        "available_images": [
                            "all_models_accuracy.png",
                            "all_models_f1score.png",
                            "accuracy_vs_f1.png",
                            "best_model_confusion_matrix.png",
                            "top10_models_ranking.png"
                        ],
                        "description": {
                            "all_models_accuracy.png": "Comparação de acurácia de todos os modelos",
                            "all_models_f1score.png": "Comparação de F1-Score de todos os modelos"
                        }
                    }
                }
            }
        }
    }
)
async def list_available_images(user: Dict[str, Any] = Depends(get_current_user)):
    """Lista todas as imagens (gráficos) disponíveis."""
    return JSONResponse(content={
        "available_images": list(IMAGE_ALLOWED),
        "description": {
            "all_models_accuracy.png": "Comparação de acurácia de todos os modelos",
            "all_models_f1score.png": "Comparação de F1-Score de todos os modelos",
            "accuracy_vs_f1.png": "Scatter plot de Accuracy vs F1-Score",
            "best_model_confusion_matrix.png": "Matriz de confusão do melhor modelo",
            "top10_models_ranking.png": "Ranking dos top 10 modelos"
        }
    })


@router.get(
    "/training/{session_id}/images/{image_name}",
    tags=["Training Metrics"],
    summary="Download de imagem de uma sessão",
    description="Retorna uma imagem (gráfico) específica de uma sessão de treinamento.",
    responses={
        200: {"description": "Imagem PNG"},
        400: {"description": "Nome de imagem inválido"},
        404: {"description": "Imagem não encontrada"}
    }
)
async def get_training_session_image(
    session_id: str,
    image_name: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Retorna uma imagem específica de uma sessão de treinamento."""
    if image_name not in IMAGE_ALLOWED:
        raise HTTPException(status_code=400, detail=f"Invalid image name. Allowed: {list(IMAGE_ALLOWED)}")

    session_path = METRICS_ROOT / session_id
    img_path = session_path / image_name

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(img_path), media_type="image/png")


@router.get(
    "/training/latest/images/{image_name}",
    tags=["Training Metrics"],
    summary="Download de imagem da sessão mais recente",
    description="Retorna uma imagem (gráfico) da sessão de treinamento mais recente.",
    responses={
        200: {"description": "Imagem PNG"},
        400: {"description": "Nome de imagem inválido"},
        404: {"description": "Imagem não encontrada ou nenhuma sessão disponível"}
    }
)
async def get_latest_training_image(
    image_name: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Retorna uma imagem da sessão de treinamento mais recente."""
    if image_name not in IMAGE_ALLOWED:
        raise HTTPException(status_code=400, detail=f"Invalid image name. Allowed: {list(IMAGE_ALLOWED)}")

    latest = _get_latest_training_session()
    if latest is None:
        raise HTTPException(status_code=404, detail="No training metrics available")

    img_path = latest / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(img_path), media_type="image/png")