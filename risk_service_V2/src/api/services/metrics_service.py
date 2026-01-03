from typing import Optional
from src.api.models.model_manager import ModelManager


class MetricsService:
    def __init__(self, manager: ModelManager = None):
        self.manager = manager or ModelManager()
        self.manager.load()

    def list_versions(self):
        return self.manager.list_versions()

    def get_metrics(self, version: Optional[str] = None):
        return self.manager.get_model_metrics(version=version)
