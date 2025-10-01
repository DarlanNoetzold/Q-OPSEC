from flask import Flask, jsonify
from controllers.risk_controller import risk_bp
from apscheduler.schedulers.background import BackgroundScheduler
from services.risk_model_service import RiskModelService
import os
import atexit

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

service = RiskModelService()

def create_app():
    ensure_dirs()
    app = Flask(__name__)
    app.register_blueprint(risk_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    scheduler = BackgroundScheduler(daemon=True)

    scheduler.add_job(
        func=service.scheduled_retrain,
        trigger="interval",
        hours=1,
        id="hourly_retrain",
        max_instances=1,
        replace_existing=True
    )

    scheduler.add_job(
        func=service.scheduled_cleanup,
        trigger="interval",
        days=3,
        id="cleanup_old_models",
        max_instances=1,
        replace_existing=True
    )

    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=False)