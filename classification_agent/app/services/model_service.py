import asyncio
import json
import os
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
import structlog

from common.ttl_cache import file_cache_key

from ..core.config import settings
from ..models.database import ModelRecord, ModelStatus

logger = structlog.get_logger()

CATEGORICAL_HINTS = ("geo", "device", "policy", "type", "status", "classification", "level")

class ModelLoadError(Exception):
    pass

class PredictionError(Exception):
    pass

class ModelService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.classes: List[str] = []
        self.required_columns: List[str] = []
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.loaded_at: Optional[datetime] = None
        self._source_key: Optional[str] = None

    async def load_latest_model(self, force: bool = False) -> bool:
        """Reload off the event loop; returns False when latest.json is unchanged and force=False."""
        try:
            return await asyncio.to_thread(self._load_latest_sync, force)
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(str(e))

    @staticmethod
    def _resolve_latest_file() -> Path:
        project_root = Path(__file__).resolve().parents[3]
        registry = Path(os.getenv("QOPSEC_MODEL_REGISTRY", project_root / "classify_scheduler" / "model_registry"))
        latest_file = registry / "latest.json"
        if not latest_file.exists():
            raise ModelLoadError(f"latest.json nao encontrado em {registry}")
        return latest_file

    @staticmethod
    def _find_artifact(registry_path: Path, tag: str) -> Path:
        for folder in sorted(registry_path.iterdir()):
            if folder.is_dir() and folder.name.startswith(str(tag)):
                artifact = next((f for f in sorted(folder.iterdir()) if f.suffix == ".pkl"), None)
                if artifact:
                    return artifact
        raise ModelLoadError("Artefato .pkl nao encontrado.")

    @staticmethod
    def _split_columns(model, preprocessor, model_info: Dict[str, Any]):
        required: List[str] = []
        categorical: List[str] = []
        numeric: List[str] = []
        step0 = model.steps[0][1] if hasattr(model, "steps") else None
        ct = step0 if step0 is not None and hasattr(step0, "transformers_") else preprocessor
        if ct is not None and hasattr(ct, "transformers_"):
            for name, _trans, cols in ct.transformers_:
                if name == "remainder":
                    continue
                required.extend(cols)
                target = categorical if any(k in name.lower() for k in ("cat", "encoder", "onehot")) else numeric
                target.extend(cols)
        if not required:
            required = list(model_info.get("required_columns") or [])
            for col in required:
                target = categorical if any(k in col.lower() for k in CATEGORICAL_HINTS) else numeric
                target.append(col)
        return required, categorical, numeric

    def _load_latest_sync(self, force: bool) -> bool:
        latest_file = self._resolve_latest_file()
        source_key = file_cache_key(latest_file)
        if not force and self.is_model_loaded() and source_key == self._source_key:
            return False

        with open(latest_file, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        tag = model_info.get("tag") or model_info.get("version")
        with open(self._find_artifact(latest_file.parent, tag), "rb") as f:
            artifact = pickle.load(f)

        model, preprocessor = (artifact.get("model"), artifact.get("preprocessor")) if isinstance(artifact, dict) else (artifact, None)
        required, categorical, numeric = self._split_columns(model, preprocessor, model_info)

        self.model, self.preprocessor = model, preprocessor
        self.required_columns, self.categorical_cols, self.numeric_cols = required, categorical, numeric
        self.classes = [str(c) for c in (model_info.get("classes") or [])]
        self.model_name = model_info.get("saved_model_name") or model_info.get("model_name")
        self.model_version = tag
        self.loaded_at = datetime.utcnow()
        self._source_key = source_key

        logger.info("Model loaded successfully", tag=tag, cats=len(categorical), nums=len(numeric))
        return True

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Describe the loaded model without touching the registry on every call."""
        if not self.is_model_loaded():
            return None
        return {
            "model_name": self.model_name,
            "saved_model_name": self.model_name,
            "model_version": self.model_version,
            "classes": list(self.classes),
            "required_columns": list(self.required_columns),
            "categorical_columns": list(self.categorical_cols),
            "numeric_columns": list(self.numeric_cols),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "meta": {},
        }

    def validate_input(self, data: Union[Dict, List]) -> pd.DataFrame:
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        
        if self.required_columns:
            for col in self.required_columns:
                if col not in df.columns:
                    if col in self.categorical_cols:
                        df[col] = "unknown"
                    else:
                        df[col] = 0.0
            
            df = df[self.required_columns]


        for col in self.categorical_cols:
            df[col] = df[col].fillna("unknown").astype(str)
        
        for col in self.numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        return df

    @staticmethod
    def _row_hashes(df: pd.DataFrame) -> List[str]:
        """Stable short hash per input row, so identical inputs can be traced/deduplicated."""
        return [
            hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]
            for row in df.to_dict(orient="records")
        ]

    def predict(self, data: Union[Dict, List]):
        try:
            if not self.is_model_loaded():
                raise PredictionError("No model loaded")
            
            df = self.validate_input(data)
            
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            
            results = []
            probs = []
            for i, pred in enumerate(predictions):
                label = self.classes[int(pred)] if (self.classes and int(pred) < len(self.classes)) else str(pred)
                results.append(label)
                probs.append(float(np.max(probabilities[i])))
            
            return results, probs, self._row_hashes(df)
            
        except Exception as e:
            logger.error("Prediction failed Detailed", error=str(e))
            raise PredictionError(f"Prediction failed: {str(e)}")

model_service = ModelService()
