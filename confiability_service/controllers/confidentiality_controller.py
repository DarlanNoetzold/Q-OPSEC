# controllers/confidentiality_controller.py
from flask import Blueprint, request, jsonify
from models.schemas import ClassifyRequest, TrainRequest, ContentConfidentiality, TrainResponse, validate_payload
from services.conf_model_service import ConfidentialityModelService
from repositories.config_repo import DATA_DIR, MODELS_DIR, read_registry, write_registry
import os
import glob

conf_bp = Blueprint("confidentiality", __name__, url_prefix="/confidentiality")
_service = ConfidentialityModelService()


@conf_bp.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    model_obj, err = validate_payload(TrainRequest, payload)
    if err:
        return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

    resp: TrainResponse = _service.train(model_obj)
    return jsonify(resp.model_dump())


@conf_bp.route("/classify", methods=["POST"])
def classify():
    payload = request.get_json(silent=True) or {}
    model_obj, err = validate_payload(ClassifyRequest, payload)
    if err:
        return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

    classified = _service.classify(model_obj)
    if classified is None:
        return jsonify({
            "error": "MODEL_NOT_READY",
            "message": "Train the model first or wait for the hourly retrain."
        }), 503

    return jsonify(classified.model_dump())


@conf_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "confidentiality"}), 200


@conf_bp.route("/cleanup/all", methods=["POST"])
def cleanup_all_conf_models():
    payload = request.get_json(silent=True) or {}
    dry_run = bool(payload.get("dry_run", False))

    files = glob.glob(os.path.join(MODELS_DIR, "conf_model_*.joblib"))
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
            registry = read_registry()
            registry["conf_models"] = []
            registry["best_conf_model"] = None
            write_registry(registry)

            _service._best_model = None
            _service._best_info = None
        except Exception as e:
            return jsonify({"error": "REGISTRY_RESET_FAILED", "message": str(e)}), 500

    return jsonify({
        "dry_run": dry_run,
        "files_found": len(files),
        "files_removed": removed if not dry_run else 0,
        "space_freed_mb": 0.0 if dry_run else round(total_size_mb, 2),
        "message": "Completed (dry_run)" if dry_run else "All confidentiality models removed and registry reset"
    }), 200


@conf_bp.route("/cleanup", methods=["POST"])
def cleanup_conf_models():
    payload = request.get_json(silent=True) or {}

    dry_run = payload.get("dry_run", True)
    keep_n = payload.get("keep_best_n", 8)
    max_age = payload.get("max_age_days", 30)
    min_acc = payload.get("min_accuracy_threshold", 0.6)

    result = _service.cleanup_old_models(
        keep_best_n=keep_n,
        max_age_days=max_age,
        min_accuracy_threshold=min_acc,
        dry_run=dry_run
    )

    return jsonify(result)


@conf_bp.route("/cleanup/recommendations", methods=["GET"])
def cleanup_recommendations():
    return jsonify(_service.get_cleanup_recommendations())