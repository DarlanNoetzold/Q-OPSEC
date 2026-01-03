from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class SingleRecord(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature name -> value mapping")


class BatchRecords(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class PredictRequest(BaseModel):
    # either provide a single record or a list of records
    single: Optional[SingleRecord] = None
    batch: Optional[BatchRecords] = None
    version: Optional[str] = None
    models: Optional[List[str]] = None


class ModelPrediction(BaseModel):
    probabilities: List[float]
    predictions: List[int]
    threshold: float
    metadata: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    version: str
    n_records: int
    models: Dict[str, ModelPrediction]