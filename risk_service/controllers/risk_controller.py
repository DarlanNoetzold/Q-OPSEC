from flask import Blueprint, request, jsonify
from models.schemas import AssessRequest, TrainRequest, RiskContext, TrainResponse, validate_payload
from services.risk_model_service import RiskModelService

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

    resp: RiskContext = _service.assess(model_obj)
    return jsonify(resp.model_dump())