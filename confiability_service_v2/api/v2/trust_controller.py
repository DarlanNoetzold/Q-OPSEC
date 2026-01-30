"""
Trust Controller - Endpoint principal da API V2
"""
from fastapi import APIRouter, HTTPException, status
from api.v2.schemas import TrustRequest, TrustResponse
from api.v2.models import TrustEvaluationModel
from core.trust_context import TrustContext
from core.orchestrator import TrustOrchestrator
from config.trust_config import TrustConfig
from storage.trust_repository import TrustRepository
from storage.fingerprint_repo import FingerprintRepository

# Singleton instances
_config = None
_trust_repo = None
_fingerprint_repo = None
_orchestrator = None


def get_orchestrator() -> TrustOrchestrator:
    """Retorna instância singleton do orchestrator"""
    global _config, _trust_repo, _fingerprint_repo, _orchestrator

    if _orchestrator is None:
        _config = TrustConfig()
        _trust_repo = TrustRepository()
        _fingerprint_repo = FingerprintRepository()
        _orchestrator = TrustOrchestrator(
            config=_config,
            trust_repo=_trust_repo,
            fingerprint_repo=_fingerprint_repo
        )

    return _orchestrator


# Router
router = APIRouter(prefix="/api/v2/trust", tags=["Trust V2"])


@router.post(
    "/evaluate",
    response_model=TrustResponse,
    status_code=status.HTTP_200_OK,
    summary="Avalia confiabilidade de informação",
    description="""
    **Information Trust Engine V2**

    Avalia a confiabilidade contextual de uma informação através de múltiplos sinais:
    - Temporal (freshness, drift)
    - Source (reliability, consistency)
    - Semantic (consistency, drift)
    - Anomaly (behavioral patterns)
    - Consistency (cross-dimensional)
    - Context (alignment, stability)

    Retorna:
    - Trust Score (0-1)
    - Trust Level (CRITICAL, LOW, MEDIUM, HIGH, VERIFIED)
    - Confidence Interval
    - Risk Flags
    - Trust DNA (fingerprint único)
    - Dimensões de confiança
    - Explicabilidade completa
    """
)
async def evaluate_trust(request: TrustRequest) -> TrustResponse:
    """
    Endpoint principal de avaliação de confiança

    Args:
        request: Payload, metadata e histórico da informação

    Returns:
        TrustResponse com score, nível, flags e explicações
    """
    try:
        # Criar contexto
        context = TrustContext(
            payload=request.payload,
            metadata=request.metadata or {},
            history=request.history or {}
        )

        # Executar avaliação
        orchestrator = get_orchestrator()
        result = orchestrator.evaluate(context)

        # Converter para response
        return TrustResponse(
            trust_score=result.trust_score,
            trust_level=result.trust_level,
            confidence_interval=result.confidence_interval,
            risk_flags=result.risk_flags,
            trust_dna=result.trust_dna_value,
            dimensions=result.dimensions,
            explainability=result.explainability
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Trust evaluation failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check do Trust Engine"
)
async def health_check():
    """Verifica se o engine está operacional"""
    try:
        orchestrator = get_orchestrator()
        return {
            "status": "healthy",
            "version": "2.0.0",
            "signals_count": len(orchestrator.signals),
            "signals": [s.name for s in orchestrator.signals]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get(
    "/config",
    status_code=status.HTTP_200_OK,
    summary="Retorna configuração atual do engine"
)
async def get_config():
    """Retorna a configuração ativa do Trust Engine"""
    try:
        orchestrator = get_orchestrator()
        config = orchestrator.config

        return {
            "trust_levels": config.trust_levels,
            "signals": {
                name: {
                    "enabled": sig.enabled,
                    "weight": sig.weight,
                    "threshold": sig.threshold
                }
                for name, sig in config.signals.items()
            },
            "storage": {
                "max_history_per_entity": config.max_history_per_entity,
                "max_history_per_source": config.max_history_per_source
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)}
        )