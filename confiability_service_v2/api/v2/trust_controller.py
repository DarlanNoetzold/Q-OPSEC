"""
Trust Engine V2 - Controller
Endpoints REST para avaliação de confiança
"""
from flask import Blueprint, request, jsonify
import time
import logging
import traceback

from bootstrap_v2 import get_trust_orchestrator
from core.trust_context import TrustContext
from api.v2.schemas import TrustEvaluationRequest, validate_request

# Blueprint
trust_v2_bp = Blueprint("trust_v2", __name__, url_prefix="/api/v2/trust")

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


# ========================================
# ENDPOINTS
# ========================================

@trust_v2_bp.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Endpoint principal: avalia confiança de informação

    Body:
    {
        "payload": {...},           # Dados a serem avaliados
        "metadata": {               # Metadados contextuais
            "source_id": "...",
            "entity_id": "...",
            "timestamp": "...",
            "data_type": "...",
            "environment": "..."
        }
    }
    """
    t0 = time.time()
    client_ip = _client_ip()

    try:
        data = request.get_json(silent=True) or {}

        logger.info(
            "POST /api/v2/trust/evaluate from %s | payload_keys=%s | metadata=%s",
            client_ip,
            list(data.get("payload", {}).keys())[:10],
            _json_preview(data.get("metadata", {}), 500)
        )

        # Validação
        req_obj, err = validate_request(TrustEvaluationRequest, data)
        if err:
            logger.warning("VALIDATION_ERROR | details=%s", _json_preview(err.errors()))
            return jsonify({
                "error": "VALIDATION_ERROR",
                "details": err.errors()
            }), 400

        # Criar contexto (SEM history)
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

        return jsonify(result.to_dict()), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/evaluate: %s\n%s", str(e), traceback.format_exc())
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@trust_v2_bp.route("/health", methods=["GET"])
def health():
    """Health check do Trust Engine V2"""
    try:
        orchestrator = get_trust_orchestrator()

        return jsonify({
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
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/health: %s", str(e))
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@trust_v2_bp.route("/config", methods=["GET"])
def get_config():
    """Retorna configuração atual do Trust Engine"""
    try:
        orchestrator = get_trust_orchestrator()
        config = orchestrator.config

        return jsonify({
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
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/config: %s", str(e))
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@trust_v2_bp.route("/history/<entity_id>", methods=["GET"])
def get_entity_history(entity_id: str):
    """Retorna histórico de avaliações de uma entidade"""
    try:
        orchestrator = get_trust_orchestrator()
        trust_repo = orchestrator.trust_repo

        # Parâmetros de paginação
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        # Buscar histórico
        history = trust_repo.get_entity_history(entity_id, limit=limit + offset)

        # Paginar
        paginated = history[offset:offset + limit]

        return jsonify({
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
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/history/%s: %s", entity_id, str(e))
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@trust_v2_bp.route("/history/source/<source_id>", methods=["GET"])
def get_source_history(source_id: str):
    """Retorna histórico de avaliações de uma fonte"""
    try:
        orchestrator = get_trust_orchestrator()
        trust_repo = orchestrator.trust_repo

        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        history = trust_repo.get_source_history(source_id, limit=limit + offset)
        paginated = history[offset:offset + limit]

        return jsonify({
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
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/history/source/%s: %s", source_id, str(e))
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@trust_v2_bp.route("/stats", methods=["GET"])
def get_stats():
    """Retorna estatísticas gerais do Trust Engine"""
    try:
        orchestrator = get_trust_orchestrator()
        trust_repo = orchestrator.trust_repo

        # Estatísticas básicas
        total_entities = len(trust_repo._entity_history)
        total_sources = len(trust_repo._source_history)

        total_evaluations = sum(
            len(events) for events in trust_repo._entity_history.values()
        )

        return jsonify({
            "total_entities": total_entities,
            "total_sources": total_sources,
            "total_evaluations": total_evaluations,
            "signals_active": len(orchestrator.signals),
            "storage": {
                "type": "in-memory",
                "entities_tracked": total_entities,
                "sources_tracked": total_sources
            }
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/stats: %s", str(e))
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@trust_v2_bp.route("/signals", methods=["GET"])
def list_signals():
    """Lista todos os sinais disponíveis"""
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

        return jsonify({
            "total": len(signals_info),
            "signals": signals_info
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /api/v2/trust/signals: %s", str(e))
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": str(e)
        }), 500