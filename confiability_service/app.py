# app.py (atualizado para incluir confidentiality service)
from flask import Flask, jsonify
from controllers.confidentiality_controller import conf_bp
from apscheduler.schedulers.background import BackgroundScheduler
from services.conf_model_service import ConfidentialityModelService
import os
import atexit


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)


# Singleton service instances
conf_service = ConfidentialityModelService()


def create_app():
    ensure_dirs()
    app = Flask(__name__)
    app.register_blueprint(conf_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "services": ["risk", "confidentiality"]}), 200

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8083, debug=False)