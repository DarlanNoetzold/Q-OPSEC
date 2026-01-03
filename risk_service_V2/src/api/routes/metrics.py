from fastapi import APIRouter, HTTPException
from typing import List, Optional

from src.api.models.model_manager import ModelManager

router = APIRouter()

_manager: ModelManager = None

def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
        _manager.load()
    return _manager


@router.get("/versions")
async def list_versions(manager: ModelManager = None):
    manager = manager or get_manager()
    versions = manager.list_versions()
    return {"versions": versions}


@router.get("/{version}")
async def get_metrics(version: str):
    manager = get_manager()
    try:
        metrics = manager.get_model_metrics(version=version)
        return {"version": version, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
