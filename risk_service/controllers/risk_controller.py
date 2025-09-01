from flask import Blueprint, request, jsonify
from models.schemas import AssessRequest, TrainRequest, RiskContext, TrainResponse, validate_payload
from services.risk_model_service import RiskModelService
# + imports para limpeza total
import os, glob
from repositories.config_repo import DATA_DIR, MODELS_DIR, write_registry

risk_bp = Blueprint("risk", __name__, url_prefix="/risk")
_service = RiskModelService()

@risk_bp.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    model_obj, err = validate_payload(TrainRequest, payload)
    if err:
        return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

    resp: TrainResponse = _service.train(model_obj)
    return jsonify(resp.model_dump())

@risk_bp.route("/assess", methods=["POST"])
def assess():
    payload = request.get_json(silent=True) or {}
    model_obj, err = validate_payload(AssessRequest, payload)
    if err:
        return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

    assessed = _service.assess(model_obj)
    if assessed is None:
        return jsonify({
            "error": "MODEL_NOT_READY",
            "message": "Train the model first or wait for the hourly retrain."
        }), 503

    return jsonify(assessed.model_dump())

@risk_bp.route("/cleanup/all", methods=["POST"])
def cleanup_all_models():
    payload = request.get_json(silent=True) or {}
    dry_run = bool(payload.get("dry_run", False))

    files = glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
    total_size_mb = 0.0
    removed = 0

    for fp in files:
        if os.path.exists(fp):
            stat = os.stat(fp)
            total_size_mb += stat.st_size / (1024 * 1024)
            if not dry_run:
                try:
                    os.remove(fp)
                    removed += 1
                except Exception as e:
                    return jsonify({"error": "DELETE_FAILED", "file": fp, "message": str(e)}), 500

    if not dry_run:
        try:
            write_registry({"models": [], "best_model": None})
            # reseta instância
            _service._best_model = None
            _service._best_info = None
        except Exception as e:
            return jsonify({"error": "REGISTRY_RESET_FAILED", "message": str(e)}), 500

    return jsonify({
        "dry_run": dry_run,
        "files_found": len(files),
        "files_removed": removed if not dry_run else 0,
        "space_freed_mb": 0.0 if dry_run else round(total_size_mb, 2),
        "message": "Completed (dry_run)" if dry_run else "All models removed and registry reset"
    }), 200