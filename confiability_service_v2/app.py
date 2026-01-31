from flask import Flask, jsonify
from controllers.confidentiality_controller import conf_bp
from controllers.dataset_controller import dataset_bp

# ✨ NOVO: Importar o controller do Trust Engine V2
from api.v2.trust_controller import router as trust_router_fastapi

from apscheduler.schedulers.background import BackgroundScheduler
from services.conf_model_service import ConfidentialityModelService
import os
import atexit


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)


conf_service = ConfidentialityModelService()


def create_app():
    ensure_dirs()
    app = Flask(__name__)

    # Blueprints existentes
    app.register_blueprint(conf_bp)
    app.register_blueprint(dataset_bp)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "services": ["risk", "confidentiality", "trust_v2"]
        }), 200

    # Scheduler existente
    scheduler = BackgroundScheduler(daemon=True)

    scheduler.add_job(
        func=conf_service.scheduled_cleanup,
        trigger="interval",
        days=3,
        id="cleanup_old_conf_models",
        max_instances=1,
        replace_existing=True
    )

    scheduler.add_job(
        func=conf_service.scheduled_retrain,
        trigger="interval",
        hours=1,
        id="hourly_conf_retrain",
        max_instances=1,
        replace_existing=True,
        start_date="2024-01-01 00:30:00"
    )

    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))

    return app


app = create_app()

# ✨ NOVO: Adicionar rotas do Trust Engine V2 (Flask-style)
from flask import request
from bootstrap_v2 import get_trust_orchestrator
from core.trust_context import TrustContext


@app.post("/api/v2/trust/evaluate")
def evaluate_trust_v2():
    """Endpoint do Trust Engine V2"""
    try:
        data = request.get_json()

        # Criar contexto
        context = TrustContext(
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            history=data.get("history", {})
        )

        # Executar avaliação
        orchestrator = get_trust_orchestrator()
        result = orchestrator.evaluate(context)

        return jsonify(result.to_dict()), 200

    except Exception as e:
        return jsonify({
            "error": "Trust evaluation failed",
            "message": str(e)
        }), 500


@app.get("/api/v2/trust/health")
def trust_health_v2():
    """Health check do Trust Engine V2"""
    try:
        orchestrator = get_trust_orchestrator()
        return jsonify({
            "status": "healthy",
            "version": "2.0.0",
            "signals_count": len(orchestrator.signals),
            "signals": [s.name for s in orchestrator.signals]
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8083, debug=False)