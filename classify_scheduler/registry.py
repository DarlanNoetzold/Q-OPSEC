import json
import pickle
from pathlib import Path
from typing import Dict, Any

class ModelRegistry:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, tag: str, pipeline, classes, cv_metrics: Dict[str, Any], holdout_metrics: Dict[str, Any], model_name: str, meta: Dict[str, Any]) -> Path:
        model_dir = self.base_dir / f"{tag}_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        info = {
            "model_name": model_name,
            "classes": classes,
            "cv_metrics": cv_metrics,
            "holdout_metrics": holdout_metrics,
            "meta": meta
        }
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(info, f, indent=2)

        return model_dir