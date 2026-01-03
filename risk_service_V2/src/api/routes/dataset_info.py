from fastapi import APIRouter, HTTPException
import yaml
from pathlib import Path

from src.api.models.model_manager import ModelManager

router = APIRouter()

_manager: ModelManager = None

def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
        _manager.load()
    return _manager


@router.get("/summary")
async def dataset_summary():
    manager = get_manager()
    try:
        summary = manager.get_dataset_summary()
        schema_path = Path("config/schema_fields.yaml")
        schema = {}
        if schema_path.exists():
            schema = yaml.safe_load(schema_path.read_text())
        return {"summary": summary, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
