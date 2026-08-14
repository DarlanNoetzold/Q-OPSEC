
from typing import Any, Dict
from pydantic import BaseModel


class DatasetInfoResponse(BaseModel):
    summary: str
    schema: Dict[str, Any]
