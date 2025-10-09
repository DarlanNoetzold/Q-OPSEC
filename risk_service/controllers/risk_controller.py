# risk_controller.py - VERSÃO ATUALIZADA COM ENDPOINTS DE MÉTRICAS

from flask import Blueprint, request, jsonify, send_from_directory
from models.schemas import AssessRequest, TrainRequest, RiskContext, TrainResponse, validate_payload
from services.risk_model_service import RiskModelService
import os, glob, time, json, traceback, logging
from repositories.config_repo import DATA_DIR, MODELS_DIR, write_registry
from pathlib import Path

risk_bp = Blueprint("risk", __name__, url_prefix="/risk")
_service = RiskModelService()

# Diretório de métricas
METRICS_DIR = Path(os.getcwd()) / "models" / "metrics"

# ---- Logging setup ----
logger = logging.getLogger("risk_controller")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---- Utils ----
MAX_PREVIEW = 2000
MAX_LIST_PREVIEW = 100


def _truncate(s: str, limit: int = MAX_PREVIEW) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else (s[:limit] + f"... (truncated {len(s) - limit} chars)")


def _json_preview(obj, limit: int = MAX_PREVIEW) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return _truncate(s, limit)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def _client_ip(req) -> str:
    return request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "-"


def _keys(d: dict):
    try:
        return list(d.keys())[:50] if isinstance(d, dict) else []
    except Exception:
        return []


def _len_safe(x) -> int:
    try:
        return len(x)
    except Exception:
        return -1


# ---- Endpoints ----

@risk_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "risk"}), 200


@risk_bp.route("/train", methods=["POST"])
def train():
    t0 = time.time()
    client_ip = _client_ip(request)
    headers_preview = {k: v for k, v in request.headers.items() if
                       k.lower() in ("content-type", "user-agent", "x-request-id")}
    payload = request.get_json(silent=True) or {}
    logger.info("POST /risk/train from %s | headers=%s | payload=%s",
                client_ip, _json_preview(headers_preview, 800), _json_preview(payload, 1200))
    try:
        model_obj, err = validate_payload(TrainRequest, payload)
        if err:
            logger.warning("VALIDATION_ERROR /risk/train | details=%s", _json_preview(err.errors()))
            return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

        resp: TrainResponse = _service.train(model_obj)
        resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else resp.__dict__
        logger.info("TRAIN OK | response=%s | elapsed_ms=%d",
                    _json_preview(resp_dump, 1200), int((time.time() - t0) * 1000))
        return jsonify(resp_dump)

    except Exception as e:
        logger.error("EXCEPTION /risk/train: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@risk_bp.route("/assess", methods=["POST"])
def assess():
    t0 = time.time()
    client_ip = _client_ip(request)
    headers_preview = {k: v for k, v in request.headers.items() if
                       k.lower() in ("content-type", "user-agent", "x-request-id")}
    payload = request.get_json(silent=True) or {}

    signals = payload.get("signals") if isinstance(payload, dict) else None
    request_id = payload.get("request_id") if isinstance(payload, dict) else None
    logger.info(
        "POST /risk/assess from %s | headers=%s | request_id=%s | signals_type=%s len=%s keys=%s | raw=%s",
        client_ip,
        _json_preview(headers_preview, 800),
        request_id,
        type(signals).__name__ if signals is not None else "None",
        _len_safe(signals) if isinstance(signals, (dict, list, tuple)) else -1,
        _keys(signals) if isinstance(signals, dict) else [],
        _json_preview(payload, 1600),
    )

    try:
        model_obj, err = validate_payload(AssessRequest, payload)
        if err:
            logger.warning("VALIDATION_ERROR /risk/assess | details=%s | elapsed_ms=%d",
                           _json_preview(err.errors()), int((time.time() - t0) * 1000))
            return jsonify({"error": "VALIDATION_ERROR", "details": err.errors()}), 400

        assessed: RiskContext | None = _service.assess(model_obj)

        if assessed is None:
            logger.info("MODEL_NOT_READY /risk/assess | request_id=%s | elapsed_ms=%d",
                        request_id, int((time.time() - t0) * 1000))
            return jsonify({
                "error": "MODEL_NOT_READY",
                "message": "Train the model first or wait for the hourly retrain."
            }), 503

        assessed_dump = assessed.model_dump() if hasattr(assessed, "model_dump") else assessed.__dict__
        highlights = {
            "risk_score": assessed_dump.get("risk_score"),
            "risk_level": assessed_dump.get("risk_level"),
            "breach_prob": assessed_dump.get("breach_prob"),
            "model_id": assessed_dump.get("model_id"),
            "alerts": assessed_dump.get("alerts"),
        }
        logger.info("ASSESS OK | request_id=%s | highlights=%s | full=%s | elapsed_ms=%d",
                    request_id, _json_preview(highlights, 400),
                    _json_preview(assessed_dump, 1200),
                    int((time.time() - t0) * 1000))

        return jsonify(assessed_dump)

    except Exception as e:
        logger.error("EXCEPTION /risk/assess | request_id=%s: %s\n%s",
                     request_id, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@risk_bp.route("/cleanup/all", methods=["POST"])
def cleanup_all_models():
    t0 = time.time()
    client_ip = _client_ip(request)
    payload = request.get_json(silent=True) or {}
    dry_run = bool(payload.get("dry_run", False))
    logger.info("POST /risk/cleanup/all from %s | dry_run=%s | raw=%s",
                client_ip, dry_run, _json_preview(payload, 800))

    try:
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
                        logger.error("DELETE_FAILED %s: %s", fp, str(e))
                        return jsonify({"error": "DELETE_FAILED", "file": fp, "message": str(e)}), 500

        if not dry_run:
            try:
                write_registry({"models": [], "best_model": None})
                _service._best_model = None
                _service._best_info = None
            except Exception as e:
                logger.error("REGISTRY_RESET_FAILED: %s\n%s", str(e), traceback.format_exc())
                return jsonify({"error": "REGISTRY_RESET_FAILED", "message": str(e)}), 500

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info("CLEANUP_ALL OK | files_found=%d removed=%d freed_mb=%.2f dry_run=%s elapsed_ms=%d",
                    len(files), removed if not dry_run else 0,
                    0.0 if dry_run else round(total_size_mb, 2),
                    dry_run, elapsed_ms)

        return jsonify({
            "dry_run": dry_run,
            "files_found": len(files),
            "files_removed": removed if not dry_run else 0,
            "space_freed_mb": 0.0 if dry_run else round(total_size_mb, 2),
            "message": "Completed (dry_run)" if dry_run else "All models removed and registry reset"
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /risk/cleanup/all: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


# ========== NOVOS ENDPOINTS DE MÉTRICAS ==========

@risk_bp.route("/metrics/latest", methods=["GET"])
def get_latest_metrics():
    """
    Retorna informações sobre a última sessão de treinamento
    """
    try:
        if not METRICS_DIR.exists():
            return jsonify({"error": "No metrics directory found"}), 404

        # Listar todas as sessões de treinamento
        sessions = sorted(
            [p for p in METRICS_DIR.iterdir() if
             p.is_dir() and (p.name.startswith("training_") or p.name.startswith("retrain_"))],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not sessions:
            return jsonify({"error": "No training sessions found"}), 404

        latest_session = sessions[0]
        session_id = latest_session.name

        # Ler o arquivo training_summary.json
        summary_file = latest_session / "training_summary.json"
        if not summary_file.exists():
            return jsonify({
                "session_id": session_id,
                "error": "Summary file not found"
            }), 404

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Listar imagens disponíveis
        images = [f.name for f in latest_session.iterdir() if f.suffix == ".png"]

        return jsonify({
            "session_id": session_id,
            "timestamp": summary_data.get("timestamp"),
            "best_model": summary_data.get("best_model"),
            "dataset_info": summary_data.get("dataset_info"),
            "statistics": summary_data.get("statistics"),
            "available_images": images,
            "image_base_url": f"/risk/metrics/{session_id}"
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /risk/metrics/latest: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@risk_bp.route("/metrics/sessions", methods=["GET"])
def list_all_sessions():
    """
    Lista todas as sessões de treinamento disponíveis
    """
    try:
        if not METRICS_DIR.exists():
            return jsonify({"error": "No metrics directory found"}), 404

        sessions = sorted(
            [p for p in METRICS_DIR.iterdir() if
             p.is_dir() and (p.name.startswith("training_") or p.name.startswith("retrain_"))],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        sessions_list = []
        for session_path in sessions:
            session_id = session_path.name
            summary_file = session_path / "training_summary.json"

            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)

                sessions_list.append({
                    "session_id": session_id,
                    "timestamp": summary_data.get("timestamp"),
                    "best_model_name": summary_data.get("best_model", {}).get("name"),
                    "best_accuracy": summary_data.get("best_model", {}).get("metrics", {}).get("accuracy"),
                    "best_f1_score": summary_data.get("best_model", {}).get("metrics", {}).get("f1"),
                    "total_models": summary_data.get("dataset_info", {}).get("total_models_trained"),
                    "url": f"/risk/metrics/{session_id}"
                })
            else:
                sessions_list.append({
                    "session_id": session_id,
                    "timestamp": None,
                    "error": "Summary not found"
                })

        return jsonify({
            "total_sessions": len(sessions_list),
            "sessions": sessions_list
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /risk/metrics/sessions: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@risk_bp.route("/metrics/<session_id>", methods=["GET"])
def get_session_metrics(session_id: str):
    """
    Retorna informações detalhadas de uma sessão específica
    """
    try:
        session_path = METRICS_DIR / session_id

        if not session_path.exists() or not session_path.is_dir():
            return jsonify({"error": f"Session {session_id} not found"}), 404

        summary_file = session_path / "training_summary.json"
        if not summary_file.exists():
            return jsonify({"error": "Summary file not found"}), 404

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Listar imagens disponíveis
        images = [f.name for f in session_path.iterdir() if f.suffix == ".png"]

        return jsonify({
            "session_id": session_id,
            "summary": summary_data,
            "available_images": images,
            "image_base_url": f"/risk/metrics/{session_id}"
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /risk/metrics/<session_id>: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@risk_bp.route("/metrics/<session_id>/<filename>", methods=["GET"])
def get_metrics_image(session_id: str, filename: str):
    """
    Serve uma imagem específica de métricas
    """
    try:
        session_path = METRICS_DIR / session_id

        if not session_path.exists() or not session_path.is_dir():
            return jsonify({"error": f"Session {session_id} not found"}), 404

        file_path = session_path / filename

        if not file_path.exists() or file_path.suffix not in [".png", ".jpg", ".jpeg", ".json"]:
            return jsonify({"error": f"File {filename} not found or invalid type"}), 404

        return send_from_directory(session_path, filename)

    except Exception as e:
        logger.error("EXCEPTION /risk/metrics/<session_id>/<filename>: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@risk_bp.route("/metrics/compare", methods=["POST"])
def compare_sessions():
    """
    Compara métricas entre múltiplas sessões de treinamento
    """
    try:
        payload = request.get_json(silent=True) or {}
        session_ids = payload.get("session_ids", [])

        if not session_ids:
            return jsonify({"error": "No session_ids provided"}), 400

        comparison = []

        for session_id in session_ids:
            session_path = METRICS_DIR / session_id

            if not session_path.exists():
                comparison.append({
                    "session_id": session_id,
                    "error": "Session not found"
                })
                continue

            summary_file = session_path / "training_summary.json"
            if not summary_file.exists():
                comparison.append({
                    "session_id": session_id,
                    "error": "Summary not found"
                })
                continue

            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            comparison.append({
                "session_id": session_id,
                "timestamp": summary_data.get("timestamp"),
                "best_model": summary_data.get("best_model", {}).get("name"),
                "metrics": summary_data.get("best_model", {}).get("metrics"),
                "statistics": summary_data.get("statistics")
            })

        return jsonify({
            "comparison": comparison
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /risk/metrics/compare: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500