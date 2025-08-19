# controllers/confidentiality_controller.py
from flask import Blueprint, request, jsonify
from models.schemas import ClassifyRequest, TrainRequest, ContentConfidentiality, TrainResponse, validate_payload
from services.conf_model_service import ConfidentialityModelService

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