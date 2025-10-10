# controllers/dataset_controller.py
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import json
import time
import logging
import traceback

dataset_bp = Blueprint("datasets", __name__, url_prefix="/datasets")

# Diretório raiz dos datasets
DATA_DIR = Path(os.getcwd()) / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("dataset_controller")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Extensões permitidas
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".jsonl", ".ndjson", ".json", ".tsv", ".txt"}


def _client_ip(req) -> str:
    return request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "-"


def _json_preview(obj, limit: int = 1000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return s if len(s) <= limit else (s[:limit] + f"... (truncated)")
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"



@dataset_bp.route("", methods=["GET"])
def list_datasets():
    t0 = time.time()
    client_ip = _client_ip(request)

    try:
        datasets = []

        for item in DATA_DIR.iterdir():
            if item.is_file():
                datasets.append({
                    "name": item.name,
                    "type": "file",
                    "path": str(item.relative_to(DATA_DIR)),
                    "size_bytes": item.stat().st_size,
                    "modified_at": item.stat().st_mtime,
                    "extension": item.suffix
                })
            elif item.is_dir():
                # Diretório (coleção de arquivos)
                files = [f.name for f in item.iterdir() if f.is_file()]
                total_size = sum(f.stat().st_size for f in item.iterdir() if f.is_file())
                modified = max((f.stat().st_mtime for f in item.iterdir() if f.is_file()),
                               default=item.stat().st_mtime)

                datasets.append({
                    "name": item.name,
                    "type": "directory",
                    "path": str(item.relative_to(DATA_DIR)),
                    "files": files,
                    "file_count": len(files),
                    "total_size_bytes": total_size,
                    "modified_at": modified
                })

        logger.info("GET /datasets from %s | found=%d | elapsed_ms=%d",
                    client_ip, len(datasets), int((time.time() - t0) * 1000))

        return jsonify({
            "total": len(datasets),
            "datasets": sorted(datasets, key=lambda x: x.get("modified_at", 0), reverse=True)
        }), 200

    except Exception as e:
        logger.error("EXCEPTION /datasets: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>", methods=["GET"])
def get_dataset_info(name: str):
    t0 = time.time()
    client_ip = _client_ip(request)

    try:
        target = DATA_DIR / secure_filename(name)

        if not target.exists():
            return jsonify({"error": "Dataset not found"}), 404

        if target.is_file():
            info = {
                "name": target.name,
                "type": "file",
                "path": str(target.relative_to(DATA_DIR)),
                "size_bytes": target.stat().st_size,
                "modified_at": target.stat().st_mtime,
                "extension": target.suffix
            }
        else:
            files = [f.name for f in target.iterdir() if f.is_file()]
            total_size = sum(f.stat().st_size for f in target.iterdir() if f.is_file())

            metadata = {}
            metadata_file = target / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            info = {
                "name": target.name,
                "type": "directory",
                "path": str(target.relative_to(DATA_DIR)),
                "files": files,
                "file_count": len(files),
                "total_size_bytes": total_size,
                "modified_at": max((f.stat().st_mtime for f in target.iterdir() if f.is_file()),
                                   default=target.stat().st_mtime),
                "metadata": metadata
            }

        logger.info("GET /datasets/%s from %s | elapsed_ms=%d",
                    name, client_ip, int((time.time() - t0) * 1000))

        return jsonify(info), 200

    except Exception as e:
        logger.error("EXCEPTION /datasets/%s: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("", methods=["POST"])
def create_dataset():
    t0 = time.time()
    client_ip = _client_ip(request)
    payload = request.get_json(silent=True) or {}

    name = payload.get("name")
    if not name:
        return jsonify({"error": "VALIDATION_ERROR", "message": "name is required"}), 400

    try:
        target = DATA_DIR / secure_filename(name)

        if target.exists():
            return jsonify({"error": "Dataset already exists"}), 409

        target.mkdir(parents=True, exist_ok=False)

        logger.info("POST /datasets from %s | created=%s | elapsed_ms=%d",
                    client_ip, name, int((time.time() - t0) * 1000))

        return jsonify({
            "status": "created",
            "name": name,
            "path": str(target.relative_to(DATA_DIR))
        }), 201

    except Exception as e:
        logger.error("EXCEPTION POST /datasets: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>", methods=["DELETE"])
def delete_dataset(name: str):
    t0 = time.time()
    client_ip = _client_ip(request)

    try:
        target = DATA_DIR / secure_filename(name)

        if not target.exists():
            return jsonify({"status": "ok", "message": "Dataset not found"}), 200

        if target.is_file():
            target.unlink()
            logger.info("DELETE /datasets/%s from %s | type=file | elapsed_ms=%d",
                        name, client_ip, int((time.time() - t0) * 1000))
        else:
            import shutil
            shutil.rmtree(target)
            logger.info("DELETE /datasets/%s from %s | type=directory | elapsed_ms=%d",
                        name, client_ip, int((time.time() - t0) * 1000))

        return jsonify({"status": "deleted", "name": name}), 200

    except Exception as e:
        logger.error("EXCEPTION DELETE /datasets/%s: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/files", methods=["POST"])
def upload_file(name: str):
    t0 = time.time()
    client_ip = _client_ip(request)

    try:
        target_dir = DATA_DIR / secure_filename(name)

        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        if not target_dir.is_dir():
            return jsonify({"error": "Target is not a directory"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_ext = Path(filename).suffix.lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": "Invalid file type",
                "allowed": list(ALLOWED_EXTENSIONS)
            }), 400

        override = request.args.get("override", "false").lower() == "true"
        dest = target_dir / filename

        if dest.exists() and not override:
            return jsonify({"error": "File already exists. Use override=true"}), 409

        file.save(str(dest))

        logger.info("POST /datasets/%s/files from %s | filename=%s | size=%d | elapsed_ms=%d",
                    name, client_ip, filename, dest.stat().st_size, int((time.time() - t0) * 1000))

        return jsonify({
            "status": "uploaded",
            "filename": filename,
            "size_bytes": dest.stat().st_size,
            "path": str(dest.relative_to(DATA_DIR))
        }), 201

    except Exception as e:
        logger.error("EXCEPTION POST /datasets/%s/files: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/files", methods=["GET"])
def list_files(name: str):
    t0 = time.time()
    client_ip = _client_ip(request)

    try:
        target_dir = DATA_DIR / secure_filename(name)

        if not target_dir.exists():
            return jsonify({"error": "Dataset not found"}), 404

        if not target_dir.is_dir():
            return jsonify({"error": "Not a directory"}), 400

        files = []
        for f in target_dir.iterdir():
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "modified_at": f.stat().st_mtime,
                    "extension": f.suffix
                })

        logger.info("GET /datasets/%s/files from %s | count=%d | elapsed_ms=%d",
                    name, client_ip, len(files), int((time.time() - t0) * 1000))

        return jsonify({
            "dataset": name,
            "file_count": len(files),
            "files": sorted(files, key=lambda x: x["modified_at"], reverse=True)
        }), 200

    except Exception as e:
        logger.error("EXCEPTION GET /datasets/%s/files: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/files/<filename>", methods=["GET"])
def download_file(name: str, filename: str):
    try:
        target_dir = DATA_DIR / secure_filename(name)

        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"error": "Dataset not found"}), 404

        file_path = target_dir / secure_filename(filename)

        if not file_path.exists() or not file_path.is_file():
            return jsonify({"error": "File not found"}), 404

        logger.info("GET /datasets/%s/files/%s | download", name, filename)

        return send_from_directory(target_dir, filename, as_attachment=True)

    except Exception as e:
        logger.error("EXCEPTION GET /datasets/%s/files/%s: %s\n%s",
                     name, filename, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/files/<filename>", methods=["DELETE"])
def delete_file(name: str, filename: str):
    t0 = time.time()
    client_ip = _client_ip(request)

    try:
        target_dir = DATA_DIR / secure_filename(name)

        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"error": "Dataset not found"}), 404

        file_path = target_dir / secure_filename(filename)

        if not file_path.exists():
            return jsonify({"status": "ok", "message": "File not found"}), 200

        file_path.unlink()

        logger.info("DELETE /datasets/%s/files/%s from %s | elapsed_ms=%d",
                    name, filename, client_ip, int((time.time() - t0) * 1000))

        return jsonify({"status": "deleted", "filename": filename}), 200

    except Exception as e:
        logger.error("EXCEPTION DELETE /datasets/%s/files/%s: %s\n%s",
                     name, filename, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/metadata", methods=["GET"])
def get_metadata(name: str):
    try:
        target_dir = DATA_DIR / secure_filename(name)

        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"error": "Dataset not found"}), 404

        metadata_file = target_dir / "metadata.json"

        if not metadata_file.exists():
            return jsonify({"metadata": {}}), 200

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return jsonify({"metadata": metadata}), 200

    except Exception as e:
        logger.error("EXCEPTION GET /datasets/%s/metadata: %s\n%s",
                     name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/metadata", methods=["POST"])
def set_metadata(name: str):
    t0 = time.time()
    client_ip = _client_ip(request)
    payload = request.get_json(silent=True) or {}

    try:
        target_dir = DATA_DIR / secure_filename(name)

        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"error": "Dataset not found"}), 404

        metadata_file = target_dir / "metadata.json"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("POST /datasets/%s/metadata from %s | elapsed_ms=%d",
                    name, client_ip, int((time.time() - t0) * 1000))

        return jsonify({"status": "ok", "message": "Metadata updated"}), 200

    except Exception as e:
        logger.error("EXCEPTION POST /datasets/%s/metadata: %s\n%s",
                     name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/preview", methods=["GET"])
def preview_dataset(name: str):
    try:
        import pandas as pd

        filename = request.args.get("file")
        n = int(request.args.get("n", 50))

        if not filename:
            return jsonify({"error": "file parameter is required"}), 400

        target_dir = DATA_DIR / secure_filename(name)
        file_path = target_dir / secure_filename(filename)

        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404

        # Ler arquivo
        ext = file_path.suffix.lower()
        if ext in [".csv", ".tsv"]:
            df = pd.read_csv(file_path, nrows=n)
        elif ext in [".jsonl", ".ndjson"]:
            df = pd.read_json(file_path, lines=True, nrows=n)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path).head(n)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        return jsonify({
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "count_returned": len(df)
        }), 200

    except ImportError:
        return jsonify({"error": "pandas not installed"}), 500
    except Exception as e:
        logger.error("EXCEPTION /datasets/%s/preview: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/schema", methods=["GET"])
def get_schema(name: str):
    try:
        import pandas as pd

        filename = request.args.get("file")

        if not filename:
            return jsonify({"error": "file parameter is required"}), 400

        target_dir = DATA_DIR / secure_filename(name)
        file_path = target_dir / secure_filename(filename)

        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404

        ext = file_path.suffix.lower()
        if ext in [".csv", ".tsv"]:
            df = pd.read_csv(file_path)
        elif ext in [".jsonl", ".ndjson"]:
            df = pd.read_json(file_path, lines=True)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "shape": list(df.shape),
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
        }

        return jsonify(schema), 200

    except ImportError:
        return jsonify({"error": "pandas not installed"}), 500
    except Exception as e:
        logger.error("EXCEPTION /datasets/%s/schema: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500


@dataset_bp.route("/<name>/stats", methods=["GET"])
def get_stats(name: str):
    try:
        import pandas as pd

        filename = request.args.get("file")

        if not filename:
            return jsonify({"error": "file parameter is required"}), 400

        target_dir = DATA_DIR / secure_filename(name)
        file_path = target_dir / secure_filename(filename)

        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404

        # Ler arquivo
        ext = file_path.suffix.lower()
        if ext in [".csv", ".tsv"]:
            df = pd.read_csv(file_path)
        elif ext in [".jsonl", ".ndjson"]:
            df = pd.read_json(file_path, lines=True)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        desc_num = df.describe(include=["number"]).to_dict()

        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        cat_freqs = {c: df[c].value_counts().head(10).to_dict() for c in cat_cols}

        return jsonify({
            "describe_numeric": desc_num,
            "categorical_freqs": cat_freqs
        }), 200

    except ImportError:
        return jsonify({"error": "pandas not installed"}), 500
    except Exception as e:
        logger.error("EXCEPTION /datasets/%s/stats: %s\n%s", name, str(e), traceback.format_exc())
        return jsonify({"error": "INTERNAL_ERROR", "message": str(e)}), 500