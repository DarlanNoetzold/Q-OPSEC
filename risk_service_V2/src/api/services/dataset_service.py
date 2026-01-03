
from src.api.models.model_manager import ModelManager


class DatasetService:
    def __init__(self, manager: ModelManager = None):
        self.manager = manager or ModelManager()
        self.manager.load()

    def get_summary_and_schema(self):
        summary = self.manager.get_dataset_summary()
        schema_path = "config/schema_fields.yaml"
        try:
            import yaml
            with open(schema_path, "r") as f:
                schema = yaml.safe_load(f)
        except Exception:
            schema = {}
        return {"summary": summary, "schema": schema}
