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
    """Lista todas as versões de modelos disponíveis."""
    versions = manager.list_versions()
    return {"versions": versions, "count": len(versions)}

@router.get("/latest")
async def get_latest_metrics(manager: ModelManager = Depends(get_manager)):
    """Retorna as métricas da versão mais recente dos modelos."""
    try:
        metrics = manager.get_model_metrics(version=None)
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics found")
        return {
            "version": manager.loaded_version,
            "metrics": metrics
        }
    except Exception as e:
        logger.exception("Error fetching latest metrics")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{version}")
async def get_metrics_by_version(version: str, manager: ModelManager = Depends(get_manager)):
    """Retorna as métricas de uma versão específica dos modelos."""
    try:
        metrics = manager.get_model_metrics(version=version)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Metrics not found for version {version}")
        return {"version": version, "metrics": metrics}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
    except Exception as e:
        logger.exception("Error fetching metrics")
        raise HTTPException(status_code=500, detail=str(e))