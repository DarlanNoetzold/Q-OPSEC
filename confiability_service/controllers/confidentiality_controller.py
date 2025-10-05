# controllers/confidentiality_controller.py
from flask import Blueprint, request, jsonify
from models.schemas import ClassifyRequest, TrainRequest, ContentConfidentiality, TrainResponse, validate_payload
from services.conf_model_service import ConfidentialityModelService
from repositories.config_repo import DATA_DIR, MODELS_DIR, read_registry, write_registry
import os, glob, time, json, traceback, logging

conf_bp = Blueprint("confidentiality", __name__, url_prefix="/confidentiality")
_service = ConfidentialityModelService()

# ---------- Logging setup ----------
logger = logging.getLogger("confidentiality_controller")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- Utils ----------
MAX_PREVIEW = 2000

def _truncate(s: str, limit: int = MAX_PREVIEW) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else (s[:limit] + f"... (truncated {len(s)-limit} chars)")

def _json_preview(obj, limit: int = MAX_PREVIEW) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return _truncate(s, limit)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"

def _client_ip(req) -> str:
    return request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "-"

def _len_safe(x) -> int:
    try:
        return len(x)
    except Exception:
        return -1

def _keys(d: dict):
    try:
        return list(d.keys())[:50] if isinstance(d, dict) else []
    except Exception:
        return []

# ---------- Endpoints ----------

@conf_bp.route("/train", methods=["POST"])
def train():
    t0 = time.time()
    client_ip = _client_ip(request)
    headers_preview = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "user-agent", "x-request-id")}
    payload = request.get_json(silent=True) or {}
    logger.info("POST /confidentiality/train from %s | headers=%s | payload=%s",
                client_ip, _json_preview(headers_preview, 800), _json_preview(payload, 1200))
    try:
        model_obj, err = validate_payload(TrainRequest, payload)
        if err:
            logger.warning("VALIDATION_ERROR /confidentiality/train | details=%s",
                           _json_preview(err.errors()))
            return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

        resp: TrainResponse = _service.train(model_obj)
        resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else resp.__dict__
        logger.info("CONF TRAIN OK | response=%s | elapsed_ms=%d",
                    _json_preview(resp_dump, 1200), int((time.time() - t0) * 1000))
        return jsonify(resp_dump)

    except Exception as e:
        logger.error("EXCEPTION /confidentiality/train: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@conf_bp.route("/classify", methods=["POST"])
def classify():
    t0 = time.time()
    client_ip = _client_ip(request)
    headers_preview = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "user-agent", "x-request-id")}
    payload = request.get_json(silent=True) or {}

    request_id = payload.get("request_id") if isinstance(payload, dict) else None
    content_pointer = payload.get("content_pointer") if isinstance(payload, dict) else None
    src = payload.get("source") if isinstance(payload, dict) else None
    dst = payload.get("destination") if isinstance(payload, dict) else None
    logger.info(
        "POST /confidentiality/classify from %s | headers=%s | request_id=%s | content_pointer_len=%s keys=%s | source_keys=%s | dest_keys=%s | raw=%s",
        client_ip,
        _json_preview(headers_preview, 800),
        request_id,
        _len_safe(content_pointer) if isinstance(content_pointer, dict) else -1,
        _keys(content_pointer) if isinstance(content_pointer, dict) else [],
        _keys(src) if isinstance(src, dict) else [],
        _keys(dst) if isinstance(dst, dict) else [],
        _json_preview(payload, 1600),
    )

    try:
        model_obj, err = validate_payload(ClassifyRequest, payload)
        if err:
            logger.warning("VALIDATION_ERROR /confidentiality/classify | details=%s | elapsed_ms=%d",
                           _json_preview(err.errors()), int((time.time() - t0) * 1000))
            return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

        classified: ContentConfidentiality | None = _service.classify(model_obj)

        if classified is None:
            logger.info("MODEL_NOT_READY /confidentiality/classify | request_id=%s | elapsed_ms=%d",
                        request_id, int((time.time() - t0) * 1000))
            return jsonify({
                "error": "MODEL_NOT_READY",
                "message": "Train the model first or wait for the hourly retrain."
            }), 503

        classified_dump = classified.model_dump() if hasattr(classified, "model_dump") else classified.__dict__
        highlights = {
            "label": classified_dump.get("label"),
            "score": classified_dump.get("score"),
            "model_id": classified_dump.get("model_id"),
            "pii_count": len(classified_dump.get("pii", []) or []),
            "secrets_count": len(classified_dump.get("secrets", []) or []),
            "cards_count": len(classified_dump.get("cards", []) or []),
        }
        logger.info("CLASSIFY OK | request_id=%s | highlights=%s | full=%s | elapsed_ms=%d",
                    request_id, _json_preview(highlights, 400),
                    _json_preview(classified_dump, 1200),
                    int((time.time() - t0) * 1000))

        return jsonify(classified_dump)

    except Exception as e:
        logger.error("EXCEPTION /confidentiality/classify | request_id=%s: %s\n%s",
                     request_id, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@conf_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "confidentiality"}), 200


@conf_bp.route("/cleanup/all", methods=["POST"])
def cleanup_all_conf_models():
    t0 = time.time()
    client_ip = _client_ip(request)
    payload = request.get_json(silent=True) or {}
    dry_run = bool(payload.get("dry_run", False))
    logger.info("POST /confidentiality/cleanup/all from %s | dry_run=%s | raw=%s",
                client_ip, dry_run, _json_preview(payload, 800))

    try:
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
                        logger.error("DELETE_FAILED %s: %s", fp, str(e))
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
                logger.error("REGISTRY_RESET_FAILED: %s\n%s", str(e), traceback.format_exc())
                return jsonify({"error": "REGISTRY_RESET_FAILED", "message": str(e)}), 500

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info("CONF CLEANUP_ALL OK | files_found=%d removed=%d freed_mb=%.2f dry_run=%s elapsed_ms=%d",
                    len(files), removed if not dry_run else 0,
                    0.0 if dry_run else round(total_size_mb, 2),
                    dry_run, elapsed_ms)

        return jsonify({
            "dry_run": dry_run,
            "files_found": len(files),
            "files_removed": removed if not dry_run else 0,
            "space_freed_mb": 0.0 if dry_run else round(total_size_mb, 2),
            "message": "Completed (dry_run)" if dry_run else "All confidentiality models removed and registry reset"
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /confidentiality/cleanup/all: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@conf_bp.route("/cleanup", methods=["POST"])
def cleanup_conf_models():
    t0 = time.time()
    client_ip = _client_ip(request)
    payload = request.get_json(silent=True) or {}

    dry_run = payload.get("dry_run", True)
    keep_n = payload.get("keep_best_n", 8)
    max_age = payload.get("max_age_days", 30)
    min_acc = payload.get("min_accuracy_threshold", 0.6)

    logger.info("POST /confidentiality/cleanup from %s | params={dry_run=%s, keep_best_n=%s, max_age_days=%s, min_acc=%.2f} | raw=%s",
                client_ip, dry_run, keep_n, max_age, min_acc, _json_preview(payload, 800))

    try:
        result = _service.cleanup_old_models(
            keep_best_n=keep_n,
            max_age_days=max_age,
            min_accuracy_threshold=min_acc,
            dry_run=dry_run
        )
        logger.info("CONF CLEANUP OK | result=%s | elapsed_ms=%d",
                    _json_preview(result, 1200), int((time.time() - t0) * 1000))
        return jsonify(result)
    except Exception as e:
        logger.error("EXCEPTION /confidentiality/cleanup: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@conf_bp.route("/cleanup/recommendations", methods=["GET"])
def cleanup_recommendations():
    try:
        recs = _service.get_cleanup_recommendations()
        logger.info("GET /confidentiality/cleanup/recommendations | result=%s",
                    _json_preview(recs, 1200))
        return jsonify(recs)
    except Exception as e:
        logger.error("EXCEPTION /confidentiality/cleanup/recommendations: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500