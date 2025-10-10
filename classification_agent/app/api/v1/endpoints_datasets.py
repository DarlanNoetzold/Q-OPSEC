# app/api/v1/endpoints_datasets.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import json
import os

from ...core.logging import get_logger
from ...core.security import get_current_user, require_auth
from ...services.model_service import model_service

logger = get_logger(__name__)
router = APIRouter()

DATASETS_ROOT = Path("C:\Projetos\Q-OPSEC\classify_scheduler\datasets").resolve()
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

class DatasetInfo(BaseModel):
    name: str
    path: str
    total_size_bytes: int
    files: List[str] = []
    modified_at: Optional[float] = None

class CreateDatasetRequest(BaseModel):
    name: str = Field(..., min_length=1)

@router.get("/datasets", response_model=List[DatasetInfo], tags=["Datasets"])
async def list_datasets(user=Depends(get_current_user)):
    datasets: List[DatasetInfo] = []
    for p in DATASETS_ROOT.iterdir():
        if p.is_dir():
            files = [f.name for f in p.iterdir() if f.is_file()]
            total = sum(f.stat().st_size for f in p.iterdir() if f.is_file())
            modified = max((f.stat().st_mtime for f in p.iterdir() if f.is_file()), default=p.stat().st_mtime)
            datasets.append(DatasetInfo(
                name=p.name,
                path=str(p),
                total_size_bytes=total,
                files=files,
                modified_at=modified
            ))
    return datasets

@router.post("/datasets", response_model=DatasetInfo, tags=["Datasets"])
async def create_dataset(payload: CreateDatasetRequest, user=Depends(require_auth)):
    target = DATASETS_ROOT / payload.name
    if target.exists():
        raise HTTPException(409, detail="Dataset already exists")
    target.mkdir(parents=True, exist_ok=False)
    return DatasetInfo(name=payload.name, path=str(target), total_size_bytes=0, files=[])

@router.get("/datasets/{name}", response_model=DatasetInfo, tags=["Datasets"])
async def get_dataset(name: str, user=Depends(get_current_user)):
    target = DATASETS_ROOT / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(404, detail="Dataset not found")
    files = [f.name for f in target.iterdir() if f.is_file()]
    total = sum(f.stat().st_size for f in target.iterdir() if f.is_file())
    modified = max((f.stat().st_mtime for f in target.iterdir() if f.is_file()), default=target.stat().st_mtime)
    return DatasetInfo(name=name, path=str(target), total_size_bytes=total, files=files, modified_at=modified)

@router.delete("/datasets/{name}", status_code=204, tags=["Datasets"])
async def delete_dataset(name: str, user=Depends(require_auth)):
    target = DATASETS_ROOT / name
    if not target.exists():
        return JSONResponse(status_code=204, content=None)
    if not target.is_dir():
        raise HTTPException(400, detail="Not a dataset directory")
    # delete directory recursively
    for p in target.rglob("*"):
        if p.is_file():
            p.unlink(missing_ok=True)
    for p in sorted(target.rglob("*"), reverse=True):
        if p.is_dir():
            p.rmdir()
    target.rmdir()
    return JSONResponse(status_code=204, content=None)

@router.post("/datasets/{name}/files", tags=["Datasets"])
async def upload_file(
    name: str,
    file: UploadFile = File(...),
    override: bool = Query(False, description="Override if file exists"),
    user=Depends(require_auth),
):
    target_dir = DATASETS_ROOT / name
    if not target_dir.exists():
        raise HTTPException(404, detail="Dataset not found")

    dest = target_dir / file.filename
    if dest.exists() and not override:
        raise HTTPException(409, detail="File already exists. Use override=true")

    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    tmp.replace(dest)

    logger.info("Dataset file uploaded", dataset=name, filename=file.filename, size_bytes=dest.stat().st_size)
    return {"status": "ok", "filename": file.filename, "size_bytes": dest.stat().st_size}

@router.get("/datasets/{name}/files", tags=["Datasets"])
async def list_files(name: str, user=Depends(get_current_user)):
    target_dir = DATASETS_ROOT / name
    if not target_dir.exists():
        raise HTTPException(404, detail="Dataset not found")
    return {"files": [f.name for f in target_dir.iterdir() if f.is_file()]}

@router.get("/datasets/{name}/files/{filename}", response_class=FileResponse, tags=["Datasets"])
async def download_file(name: str, filename: str, user=Depends(get_current_user)):
    path = (DATASETS_ROOT / name / filename).resolve()
    if not path.exists() or not path.is_file() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    return FileResponse(path, filename=filename)

@router.delete("/datasets/{name}/files/{filename}", status_code=204, tags=["Datasets"])
async def delete_file(name: str, filename: str, user=Depends(require_auth)):
    path = (DATASETS_ROOT / name / filename).resolve()
    if not path.exists() or not path.is_file() or DATASETS_ROOT not in path.parents:
        return JSONResponse(status_code=204, content=None)
    path.unlink(missing_ok=True)
    return JSONResponse(status_code=204, content=None)

@router.post("/datasets/{name}/metadata", tags=["Datasets"])
async def set_metadata(name: str, metadata: Dict[str, Any], user=Depends(require_auth)):
    target = DATASETS_ROOT / name
    if not target.exists():
        raise HTTPException(404, detail="Dataset not found")
    with (target / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return {"status": "ok"}

@router.get("/datasets/{name}/metadata", tags=["Datasets"])
async def get_metadata(name: str, user=Depends(get_current_user)):
    target = DATASETS_ROOT / name
    if not target.exists():
        raise HTTPException(404, detail="Dataset not found")
    meta_path = target / "metadata.json"
    if not meta_path.exists():
        return {"metadata": {}}
    with meta_path.open("r", encoding="utf-8") as f:
        return {"metadata": json.load(f)}

# Preview, schema e stats (requer pandas se for CSV/Parquet)
def _read_table(path: Path, nrows: Optional[int] = None):
    import pandas as pd
    if path.suffix.lower() in [".csv", ".tsv"]:
        return pd.read_csv(path, nrows=nrows)
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path) if nrows is None else pd.read_parquet(path).head(nrows)
    if path.suffix.lower() in [".jsonl", ".ndjson"]:
        return pd.read_json(path, lines=True, nrows=nrows)
    raise ValueError(f"Unsupported file type: {path.suffix}")

@router.get("/datasets/{name}/preview", tags=["Datasets"])
async def preview(name: str, file: str = Query(...), n: int = Query(50, ge=1, le=500), user=Depends(get_current_user)):
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        df = _read_table(path, nrows=n)
        return {
            "columns": list(df.columns),
            "rows": df.head(n).to_dict(orient="records"),
            "count_returned": min(n, len(df))
        }
    except Exception as e:
        logger.error("Preview error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Preview failed: {e}")

@router.get("/datasets/{name}/schema", tags=["Datasets"])
async def schema(name: str, file: str = Query(...), user=Depends(get_current_user)):
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        df = _read_table(path)
        inferred = {col: str(df[col].dtype) for col in df.columns}
        return {
            "columns": list(df.columns),
            "dtypes": inferred,
            "shape": list(df.shape),
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
        }
    except Exception as e:
        logger.error("Schema error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Schema inference failed: {e}")

@router.get("/datasets/{name}/stats", tags=["Datasets"])
async def stats(name: str, file: str = Query(...), user=Depends(get_current_user)):
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        import pandas as pd
        df = _read_table(path)
        desc_num = df.describe(include=["number"]).to_dict()
        # frequências para categóricas (top 10)
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        cat_freqs = {c: df[c].value_counts().head(10).to_dict() for c in cat_cols}
        return {"describe_numeric": desc_num, "categorical_freqs": cat_freqs}
    except Exception as e:
        logger.error("Stats error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Stats failed: {e}")

@router.post("/datasets/{name}/validate", tags=["Datasets"])
async def validate_dataset(name: str, file: str = Query(...), user=Depends(get_current_user)):
    if not model_service.is_model_loaded():
        raise HTTPException(503, detail="Model not loaded")
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        df = _read_table(path, nrows=200)  # sample
        required = set(model_service.get_required_columns() or [])
        missing = [c for c in required if c not in df.columns]
        extra = [c for c in df.columns if c not in required] if required else []
        valid = len(missing) == 0
        return {"valid": valid, "missing_columns": missing, "extra_columns": extra, "required": list(required)}
    except Exception as e:
        logger.error("Validation error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Validation failed: {e}")