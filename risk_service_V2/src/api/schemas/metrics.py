from typing import Any, Dict
from pydantic import BaseModel


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]