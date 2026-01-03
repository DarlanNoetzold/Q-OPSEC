from typing import List, Dict, Any, Optional
from src.api.models.model_manager import ModelManager


class PredictionService:
    def __init__(self, manager: Optional[ModelManager] = None):
        self.manager = manager or ModelManager()
        self.manager.load()

    def predict(self, records: List[Dict[str, Any]], version: Optional[str] = None, models: Optional[List[str]] = None):
        if version:
            self.manager.load(version)
        return self.manager.predict(records, model_names=models)
