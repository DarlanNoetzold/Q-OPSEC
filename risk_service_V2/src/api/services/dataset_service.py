from src.api.models.model_manager import ModelManager
from src.common.logger import logger

class DatasetService:
    def __init__(self, manager: ModelManager = None):
        self.manager = manager or ModelManager()
        try:
            self.manager.load()
        except Exception as e:
            logger.exception("Failed to load models in DatasetService")

    def get_summary_and_schema(self):
        summary = self.manager.get_dataset_summary()
        schema_path = "config/schema_fields.yaml"
        try:
            import yaml
            with open(schema_path, "r") as f:
                schema = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load schema file {schema_path}: {e}")
            schema = {}
        return {"summary": summary, "schema": schema}