import json
import pickle
from pathlib import Path
from typing import Dict, Any, List


class ModelRegistry:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
            self,
            tag: str,
            pipeline,
            classes: List[str],  # Classes já ordenadas por severidade
            cv_metrics: Dict[str, Any],
            holdout_metrics: Dict[str, Any],
            model_name: str,
            meta: Dict[str, Any],
            candidates: List[Dict[str, Any]] = None
    ) -> Path:
        model_dir = self.base_dir / f"{tag}_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        info = {
            "saved_model_name": model_name,
            "artifact_path": str(model_path.resolve()),
            "classes": classes,  # Classes ordenadas por severidade
            "cv_metrics": cv_metrics,
            "holdout_metrics": holdout_metrics,
            "meta": meta
        }
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(info, f, indent=2)

        if candidates:
            with open(model_dir / "candidates.json", "w") as f:
                json.dump({"candidates": candidates}, f, indent=2)

        # Ponteiro para o último/best - INCLUIR CLASSES NO LATEST.JSON
        latest_info = {
            "saved_model_name": model_name,
            "artifact_path": str(model_path.resolve()),
            "tag": tag,
            "dir": str(model_dir.resolve()),
            "classes": classes,  # IMPORTANTE: incluir classes ordenadas aqui também
            "meta": meta  # Incluir meta com required_columns
        }

        with open(self.base_dir / "latest.json", "w") as f:
            json.dump(latest_info, f, indent=2)

        print(f"[REGISTRY] Saved classes in order: {classes}")
        return model_dir