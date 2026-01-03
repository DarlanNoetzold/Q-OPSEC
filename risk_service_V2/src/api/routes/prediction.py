from fastapi import APIRouter, HTTPException, Depends
from typing import List

from src.api.schemas.prediction import PredictRequest, PredictResponse
from src.api.models.model_manager import ModelManager

router = APIRouter()

_manager: ModelManager = None

def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
        _manager.load()
    return _manager


@router.post("/", response_model=PredictResponse)
async def predict(req: PredictRequest, manager: ModelManager = Depends(get_manager)):
    # prepare records
    if req.single is None and req.batch is None:
        raise HTTPException(status_code=400, detail="Provide either single or batch payload")

    if req.single:
        records = [req.single.features]
    else:
        records = req.batch.records

    try:
        if req.version:
            manager.load(req.version)
        result = manager.predict(records, model_names=req.models)

        # Convert models dict to PredictResponse compatible structure
        models_resp = {}
        for k, v in result.get("models", {}).items():
            if "error" in v:
                # include as empty model prediction with error in metadata
                models_resp[k] = {
                    "probabilities": [],
                    "predictions": [],
                    "threshold": v.get("threshold", 0.5),
                    "metadata": {"error": v.get("error")},
                }
            else:
                models_resp[k] = {
                    "probabilities": v.get("probabilities", []),
                    "predictions": v.get("predictions", []),
                    "threshold": v.get("threshold", 0.5),
                    "metadata": v.get("metadata", {}),
                }

        response = {
            "version": result.get("version"),
            "n_records": result.get("n_records"),
            "models": models_resp,
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))