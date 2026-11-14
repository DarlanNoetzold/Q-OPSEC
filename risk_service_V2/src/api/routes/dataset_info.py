import asyncio

from fastapi import APIRouter, HTTPException, Depends

from src.api.dependencies import get_manager, load_schema
from src.api.models.model_manager import ModelManager

router = APIRouter()


@router.get("/summary")
async def dataset_summary(manager: ModelManager = Depends(get_manager)):
    try:
        summary = await asyncio.to_thread(manager.get_dataset_summary)
        schema = await asyncio.to_thread(load_schema)
        return {"summary": summary, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
