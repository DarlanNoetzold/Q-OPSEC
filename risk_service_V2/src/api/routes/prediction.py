from typing import List, Dict, Any, Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_manager, get_versioned_manager
from src.api.models.model_manager import ModelManager
from src.common.logger import logger

router = APIRouter()

class PredictSingle(BaseModel):
    features: Dict[str, Any]

class PredictRequest(BaseModel):
    single: Optional[PredictSingle] = None
    batch: Optional[Dict[str, List[Dict[str, Any]]]] = None
    models: Optional[List[str]] = None
    version: Optional[str] = None
    request_id: Optional[str] = None

__all__ = ["router", "get_manager"]


def _ensure_feature_list(manager: ModelManager) -> None:
    if isinstance(manager.feature_names, list):
        return
    try:
        if isinstance(manager.feature_names, dict) and "all_features" in manager.feature_names:
            manager.feature_names = manager.feature_names["all_features"]
        else:
            manager.feature_names = list(manager.feature_names or [])
    except Exception:
        manager.feature_names = []


@router.post("/")
async def predict(req: PredictRequest, manager: ModelManager = Depends(get_manager)):
    try:
        if req.single:
            records = [req.single.features]
        elif req.batch and "records" in req.batch:
            records = req.batch["records"]
        else:
            raise HTTPException(status_code=400, detail="Request must include 'single' or 'batch' records.")

        normalized_records = []
        for r in records:
            if isinstance(r, dict) and "features" in r and isinstance(r["features"], dict):
                normalized_records.append(r["features"])
            else:
                normalized_records.append(r)

        if req.version:
            try:
                manager = await asyncio.to_thread(get_versioned_manager, manager, req.version)
            except Exception as e:
                logger.exception(f"Failed to load model version {req.version}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model version {req.version}: {e}")

        _ensure_feature_list(manager)
        logger.info(f"[{req.request_id}] Predict endpoint called: n_records={len(normalized_records)}; expected_features={len(manager.feature_names)}")
        result = await asyncio.to_thread(manager.predict, normalized_records, model_names=req.models)
        logger.info(f"[{req.request_id}] RISK_SERVICE | SUCCESS | Prediction completed")

        if isinstance(result, dict):
            result["request_id"] = req.request_id
            
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in predict endpoint")
        raise HTTPException(status_code=500, detail=str(e))