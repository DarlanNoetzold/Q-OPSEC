from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

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

_manager: Optional[ModelManager] = None

def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
        try:
            _manager.load()
        except Exception as e:
            logger.exception(f"Error loading models in get_manager: {e}")
            raise
    return _manager

@router.post("/")
async def predict(req: PredictRequest, manager: ModelManager = Depends(get_manager)):
    try:
        # Build input records
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

        # Optionally load specific version
        if req.version:
            try:
                manager.load(req.version)
            except Exception as e:
                logger.exception(f"Failed to load model version {req.version}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model version {req.version}: {e}")

        # Defensive coercion: ensure feature_names is a list
        if not isinstance(manager.feature_names, list):
            logger.debug(f"Coercing manager.feature_names from {type(manager.feature_names)} to list.")
            try:
                if isinstance(manager.feature_names, dict) and "all_features" in manager.feature_names:
                    manager.feature_names = manager.feature_names["all_features"]
                else:
                    manager.feature_names = list(manager.feature_names or [])
            except Exception:
                manager.feature_names = []

        logger.info(f"Predict endpoint called: n_records={len(normalized_records)}; expected_features={len(manager.feature_names)}")
        result = manager.predict(normalized_records, model_names=req.models)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in predict endpoint")
        raise HTTPException(status_code=500, detail=str(e))