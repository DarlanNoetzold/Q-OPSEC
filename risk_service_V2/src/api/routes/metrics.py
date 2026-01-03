from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from src.api.models.model_manager import ModelManager
from src.common.logger import logger

router = APIRouter()

_manager: Optional[ModelManager] = None

def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
        try:
            _manager.load()
        except Exception as e:
            logger.exception("Failed to load models in get_manager (metrics)")
            raise
    return _manager

@router.get("/versions")
async def list_versions(manager: ModelManager = Depends(get_manager)):
    versions = manager.list_versions()
    return {"versions": versions}

@router.get("/{version}")
async def get_metrics(version: str, manager: ModelManager = Depends(get_manager)):
    try:
        metrics = manager.get_model_metrics(version=version)
        return {"version": version, "metrics": metrics}
    except Exception as e:
        logger.exception("Error fetching metrics")
        raise HTTPException(status_code=500, detail=str(e))