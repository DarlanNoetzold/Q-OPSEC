"""
Trust Engine V2 - Controller
Endpoints REST para avaliação de confiança com documentação OpenAPI
"""
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields
import time
import logging
import traceback

from bootstrap_v2 import get_trust_orchestrator
from core.trust_context import TrustContext
from api.v2.schemas import TrustEvaluationRequest, validate_request

# Blueprint com flask-smorest
trust_v2_bp = Blueprint(
    "trust_v2",
    __name__,
    url_prefix="/api/v2/trust",
    description="Trust Engine V2 - Avaliação contextual de confiança em informações"
)

# Logging
logger = logging.getLogger("trust_v2_controller")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _client_ip() -> str:
    """Extrai IP do cliente"""
    return request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "-"


def _json_preview(obj, limit: int = 1000) -> str:
    """Preview de JSON para logs"""
    import json
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return s if len(s) <= limit else (s[:limit] + "...")
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


# ========== SCHEMAS PARA DOCUMENTAÇÃO ==========

class MetadataSchema(Schema):
    """Metadados contextuais da avaliação"""
    source_id = fields.Str(required=True, metadata={"description": "ID da fonte de dados", "example": "security_system_1"})
    entity_id = fields.Str(metadata={"description": "ID da entidade avaliada", "example": "user_12345"})
    timestamp = fields.Str(metadata={"description": "Timestamp ISO 8601", "example": "2024-01-15T10:30:00Z"})
    data_type = fields.Str(metadata={"description": "Tipo de dado", "example": "security_event"})
    environment = fields.Str(metadata={"description": "Ambiente de execução", "example": "production"})


class EvaluateRequestSchema(Schema):
    """Request para avaliação de confiança"""
    payload = fields.Dict(
        required=True,
        metadata={
            "description": "Dados a serem avaliados",
            "example": {"claim": "User reported suspicious activity", "details": {"ip": "192.168.1.1"}}
        }
    )
    metadata = fields.Nested(
        MetadataSchema,
        required=True,
        metadata={"description": "Metadados contextuais"}
    )


class SignalResultSchema(Schema):
    """Resultado de um sinal individual"""
    name = fields.Str(metadata={"description": "Nome do sinal"})
    score = fields.Float(metadata={"description": "Score do sinal (0-1)"})
    weight = fields.Float(metadata={"description": "Peso do sinal"})
    confidence = fields.Float(metadata={"description": "Confiança do sinal"})


class EvaluateResponseSchema(Schema):
    """Response da avaliação de confiança"""
    trust_score = fields.Float(required=True, metadata={"description": "Score de confiança (0-1)", "example": 0.85})
    trust_level = fields.Str(required=True, metadata={"description": "Nível de confiança", "example": "HIGH"})
    risk_flags = fields.List(fields.Str(), metadata={"description": "Flags de risco identificadas"})
    signals = fields.List(fields.Nested(SignalResultSchema), metadata={"description": "Resultados dos sinais"})
    timestamp = fields.Str(metadata={"description": "Timestamp da avaliação"})
    entity_id = fields.Str(metadata={"description": "ID da entidade avaliada"})
    source_id = fields.Str(metadata={"description": "ID da fonte"})


class HealthResponseSchema(Schema):
    """Response do health check"""
    status = fields.Str(metadata={"description": "Status do serviço", "example": "healthy"})
    version = fields.Str(metadata={"description": "Versão da API", "example": "2.0.0"})
    engine = fields.Str(metadata={"description": "Nome do engine"})
    signals = fields.Dict(metadata={"description": "Informações sobre sinais ativos"})
    storage = fields.Dict(metadata={"description": "Informações sobre storage"})


class ConfigResponseSchema(Schema):
    """Response da configuração"""
    version = fields.Str()
    trust_levels = fields.Dict()
    signals = fields.Dict()
    storage = fields.Dict()
    performance = fields.Dict()


class HistoryEventSchema(Schema):
    """Evento de histórico"""
    timestamp = fields.Str()
    trust_score = fields.Float()
    trust_level = fields.Str()
    source_id = fields.Str()
    entity_id = fields.Str()
    data_type = fields.Str()
    risk_flags = fields.List(fields.Str())


class HistoryResponseSchema(Schema):
    """Response de histórico"""
    entity_id = fields.Str(metadata={"description": "ID da entidade"})
    source_id = fields.Str(metadata={"description": "ID da fonte"})
    total = fields.Int(metadata={"description": "Total de eventos"})
    limit = fields.Int()
    offset = fields.Int()
    history = fields.List(fields.Nested(HistoryEventSchema))


class StatsResponseSchema(Schema):
    """Response de estatísticas"""
    total_entities = fields.Int()
    total_sources = fields.Int()
    total_evaluations = fields.Int()
    signals_active = fields.Int()
    storage = fields.Dict()


class SignalInfoSchema(Schema):
    """Informações de um sinal"""
    name = fields.Str()
    enabled = fields.Bool()
    weight = fields.Float()
    threshold = fields.Float()
    description = fields.Str()


class SignalsListResponseSchema(Schema):
    """Response da lista de sinais"""
    total = fields.Int()
    signals = fields.List(fields.Nested(SignalInfoSchema))


class ErrorResponseSchema(Schema):
    """Response de erro"""
    error = fields.Str(metadata={"description": "Código do erro"})
    message = fields.Str(metadata={"description": "Mensagem de erro"})
    details = fields.Dict(metadata={"description": "Detalhes adicionais"})


# ========== ENDPOINTS ==========

@trust_v2_bp.route("/evaluate")
class TrustEvaluate(MethodView):

    @trust_v2_bp.arguments(EvaluateRequestSchema)
    @trust_v2_bp.response(200, EvaluateResponseSchema)
    @trust_v2_bp.alt_response(400, schema=ErrorResponseSchema, description="Erro de validação")
    @trust_v2_bp.alt_response(500, schema=ErrorResponseSchema, description="Erro interno")
    @trust_v2_bp.doc(tags=["Trust Evaluation"])
    def post(self, data):
        """
        Avalia confiança de informação

        Realiza avaliação multi-dimensional de trustworthiness baseada em:
        - Sinais temporais (freshness, consistency)
        - Sinais de fonte (reputation, history)
        - Sinais semânticos (coherence, sentiment)
        - Detecção de anomalias
        - Análise de consistência contextual
        """
        t0 = time.time()
        client_ip = _client_ip()

        try:
            logger.info(
                "POST /api/v2/trust/evaluate from %s | payload_keys=%s | metadata=%s",
                client_ip,
                list(data.get("payload", {}).keys())[:10],
                _json_preview(data.get("metadata", {}), 500)
            )

            # Validação (já feita pelo marshmallow, mas mantemos compatibilidade)
            req_obj, err = validate_request(TrustEvaluationRequest, data)
            if err:
                logger.warning("VALIDATION_ERROR | details=%s", _json_preview(err.errors()))
                abort(400, message="VALIDATION_ERROR", errors=err.errors())

            # Criar contexto
            context = TrustContext(
                payload=req_obj.payload,
                metadata=req_obj.metadata
            )

            # Executar avaliação
            orchestrator = get_trust_orchestrator()
            result = orchestrator.evaluate(context)

            elapsed_ms = int((time.time() - t0) * 1000)

            logger.info(
                "TRUST EVALUATION OK | trust_score=%.3f | trust_level=%s | risk_flags=%d | elapsed_ms=%d",
                result.trust_score,
                result.trust_level,
                len(result.risk_flags),
                elapsed_ms
            )

            return result.to_dict()

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/evaluate: %s\n%s", str(e), traceback.format_exc())
            abort(500, message="INTERNAL_ERROR", error=str(e))


@trust_v2_bp.route("/health")
class TrustHealth(MethodView):

    @trust_v2_bp.response(200, HealthResponseSchema)
    @trust_v2_bp.alt_response(503, schema=ErrorResponseSchema, description="Serviço indisponível")
    @trust_v2_bp.doc(tags=["Health & Config"])
    def get(self):
        """
        Health check do Trust Engine V2

        Retorna status do serviço, sinais ativos e informações de storage.
        """
        try:
            orchestrator = get_trust_orchestrator()

            return {
                "status": "healthy",
                "version": "2.0.0",
                "engine": "Trust Engine V2",
                "signals": {
                    "total": len(orchestrator.signals),
                    "enabled": [s.name for s in orchestrator.signals],
                    "count_by_category": {
                        "temporal": len([s for s in orchestrator.signals if "temporal" in s.name.lower()]),
                        "source": len([s for s in orchestrator.signals if "source" in s.name.lower()]),
                        "semantic": len([s for s in orchestrator.signals if "semantic" in s.name.lower()]),
                        "anomaly": len([s for s in orchestrator.signals if "anomaly" in s.name.lower()]),
                        "consistency": len([s for s in orchestrator.signals if "consistency" in s.name.lower()]),
                        "context": len([s for s in orchestrator.signals if "context" in s.name.lower()])
                    }
                },
                "storage": {
                    "trust_repository": "in-memory",
                    "fingerprint_repository": "in-memory",
                    "trust_graph": "stub"
                }
            }

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/health: %s", str(e))
            abort(503, message="Service unhealthy", error=str(e))


@trust_v2_bp.route("/config")
class TrustConfig(MethodView):

    @trust_v2_bp.response(200, ConfigResponseSchema)
    @trust_v2_bp.alt_response(500, schema=ErrorResponseSchema)
    @trust_v2_bp.doc(tags=["Health & Config"])
    def get(self):
        """
        Retorna configuração atual do Trust Engine

        Inclui configuração de sinais, storage e performance.
        """
        try:
            orchestrator = get_trust_orchestrator()
            config = orchestrator.config

            return {
                "version": "2.0.0",
                "trust_levels": config.trust_levels,
                "signals": {
                    name: {
                        "enabled": sig.enabled,
                        "weight": sig.weight,
                        "threshold": sig.threshold,
                        "params": sig.params
                    }
                    for name, sig in config.signals.items()
                },
                "storage": {
                    "max_history_per_entity": config.max_history_per_entity,
                    "max_history_per_source": config.max_history_per_source,
                    "history_retention_days": config.history_retention_days
                },
                "performance": {
                    "enable_caching": config.enable_caching,
                    "cache_ttl_seconds": config.cache_ttl_seconds,
                    "max_concurrent_evaluations": config.max_concurrent_evaluations
                }
            }

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/config: %s", str(e))
            abort(500, message="INTERNAL_ERROR", error=str(e))


@trust_v2_bp.route("/history/<entity_id>")
class EntityHistory(MethodView):

    @trust_v2_bp.response(200, HistoryResponseSchema)
    @trust_v2_bp.alt_response(500, schema=ErrorResponseSchema)
    @trust_v2_bp.doc(tags=["History"], params={"entity_id": {"description": "ID da entidade", "example": "user_12345"}})
    def get(self, entity_id: str):
        """
        Retorna histórico de avaliações de uma entidade

        Suporta paginação via query params `limit` e `offset`.
        """
        try:
            orchestrator = get_trust_orchestrator()
            trust_repo = orchestrator.trust_repo

            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))

            history = trust_repo.get_entity_history(entity_id, limit=limit + offset)
            paginated = history[offset:offset + limit]

            return {
                "entity_id": entity_id,
                "total": len(history),
                "limit": limit,
                "offset": offset,
                "history": [
                    {
                        "timestamp": event.get("timestamp"),
                        "trust_score": event.get("trust_score"),
                        "trust_level": event.get("trust_level"),
                        "source_id": event.get("source_id"),
                        "data_type": event.get("data_type"),
                        "risk_flags": event.get("risk_flags", [])
                    }
                    for event in paginated
                ]
            }

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/history/%s: %s", entity_id, str(e))
            abort(500, message="INTERNAL_ERROR", error=str(e))


@trust_v2_bp.route("/history/source/<source_id>")
class SourceHistory(MethodView):

    @trust_v2_bp.response(200, HistoryResponseSchema)
    @trust_v2_bp.alt_response(500, schema=ErrorResponseSchema)
    @trust_v2_bp.doc(tags=["History"], params={"source_id": {"description": "ID da fonte", "example": "security_system_1"}})
    def get(self, source_id: str):
        """
        Retorna histórico de avaliações de uma fonte

        Suporta paginação via query params `limit` e `offset`.
        """
        try:
            orchestrator = get_trust_orchestrator()
            trust_repo = orchestrator.trust_repo

            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))

            history = trust_repo.get_source_history(source_id, limit=limit + offset)
            paginated = history[offset:offset + limit]

            return {
                "source_id": source_id,
                "total": len(history),
                "limit": limit,
                "offset": offset,
                "history": [
                    {
                        "timestamp": event.get("timestamp"),
                        "trust_score": event.get("trust_score"),
                        "trust_level": event.get("trust_level"),
                        "entity_id": event.get("entity_id"),
                        "data_type": event.get("data_type")
                    }
                    for event in paginated
                ]
            }

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/history/source/%s: %s", source_id, str(e))
            abort(500, message="INTERNAL_ERROR", error=str(e))


@trust_v2_bp.route("/stats")
class TrustStats(MethodView):

    @trust_v2_bp.response(200, StatsResponseSchema)
    @trust_v2_bp.alt_response(500, schema=ErrorResponseSchema)
    @trust_v2_bp.doc(tags=["Statistics"])
    def get(self):
        """
        Retorna estatísticas gerais do Trust Engine

        Inclui contadores de entidades, fontes e avaliações totais.
        """
        try:
            orchestrator = get_trust_orchestrator()
            trust_repo = orchestrator.trust_repo

            total_entities = len(trust_repo._entity_history)
            total_sources = len(trust_repo._source_history)

            total_evaluations = sum(
                len(events) for events in trust_repo._entity_history.values()
            )

            return {
                "total_entities": total_entities,
                "total_sources": total_sources,
                "total_evaluations": total_evaluations,
                "signals_active": len(orchestrator.signals),
                "storage": {
                    "type": "in-memory",
                    "entities_tracked": total_entities,
                    "sources_tracked": total_sources
                }
            }

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/stats: %s", str(e))
            abort(500, message="INTERNAL_ERROR", error=str(e))


@trust_v2_bp.route("/signals")
class TrustSignals(MethodView):

    @trust_v2_bp.response(200, SignalsListResponseSchema)
    @trust_v2_bp.alt_response(500, schema=ErrorResponseSchema)
    @trust_v2_bp.doc(tags=["Health & Config"])
    def get(self):
        """
        Lista todos os sinais disponíveis

        Retorna informações sobre cada sinal: nome, status, peso e threshold.
        """
        try:
            orchestrator = get_trust_orchestrator()

            signals_info = []
            for signal in orchestrator.signals:
                config = orchestrator.config.get_signal_config(signal.name)

                signals_info.append({
                    "name": signal.name,
                    "enabled": config.enabled if config else True,
                    "weight": config.weight if config else 1.0,
                    "threshold": config.threshold if config else 0.5,
                    "description": signal.__class__.__doc__ or "No description"
                })

            return {
                "total": len(signals_info),
                "signals": signals_info
            }

        except Exception as e:
            logger.error("EXCEPTION /api/v2/trust/signals: %s", str(e))
            abort(500, message="INTERNAL_ERROR", error=str(e))
