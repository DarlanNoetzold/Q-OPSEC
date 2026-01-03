from typing import List, Dict, Any, Optional
from src.api.models.model_manager import ModelManager
from src.common.logger import logger

class PredictionService:
    def __init__(self, manager: Optional[ModelManager] = None):
        self.manager = manager or ModelManager()
        try:
            self.manager.load()
        except Exception as e:
            logger.exception("Failed to load models in PredictionService")

    def predict(self, records: List[Dict[str, Any]], version: Optional[str] = None, models: Optional[List[str]] = None):
        if version:
            try:
                self.manager.load(version)
            except Exception as e:
                logger.exception(f"Failed to load version {version} in PredictionService: {e}")
                raise
        return self.manager.predict(records, model_names=models)