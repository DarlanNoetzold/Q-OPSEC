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

from ..core.config import settings
from ..models.database import ModelRecord, ModelStatus

logger = structlog.get_logger()

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

    async def load_latest_model(self, force: bool = False) -> bool:
        try:
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parents[3] 
            
            possible_latest = [
                project_root / "classify_scheduler" / "model_registry" / "latest.json",
                Path("/home/umbrel/projetos/Q-OPSEC/classify_scheduler/model_registry/latest.json")
            ]
            
            latest_file = next((p for p in possible_latest if p.exists()), None)
            if not latest_file:
                raise ModelLoadError("latest.json nao encontrado.")

            with open(latest_file, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            tag = model_info.get("tag") or model_info.get("version")
            registry_path = latest_file.parent
            
            model_path = None
            if os.path.exists(registry_path):
                for entry in os.listdir(registry_path):
                    if entry.startswith(str(tag)) and os.path.isdir(registry_path / entry):
                        folder_path = registry_path / entry
                        for file in os.listdir(folder_path):
                            if file.endswith(".pkl"):
                                model_path = folder_path / file
                                break
                    if model_path: break

            if not model_path: raise ModelLoadError("Artefato .pkl nao encontrado.")

            with open(model_path, "rb") as f:
                artifact = pickle.load(f)

            if isinstance(artifact, dict):
                self.model = artifact.get("model")
                self.preprocessor = artifact.get("preprocessor")
            else:
                self.model = artifact
                self.preprocessor = None


            self.categorical_cols = []
            self.numeric_cols = []
            self.required_columns = []


            try:

                step0 = self.model.steps[0][1] if hasattr(self.model, "steps") else None
                ct = step0 if step0 and hasattr(step0, "transformers_") else self.preprocessor

                if ct and hasattr(ct, "transformers_"):
                   for name, trans, cols in ct.transformers_:
                       if name != 'remainder':
                           self.required_columns.extend(cols)
                           if any(keyword in name.lower() for keyword in ['cat', 'encoder', 'onehot']):
                               self.categorical_cols.extend(cols)
                           else:
                               self.numeric_cols.extend(cols)
            except:
                pass

            if not self.required_columns:
                self.required_columns = list(model_info.get("required_columns") or [])

                for col in self.required_columns:
                    if any(k in col.lower() for k in ['geo', 'device', 'policy', 'type', 'status', 'classification', 'level']):
                        self.categorical_cols.append(col)
                    else:
                        self.numeric_cols.append(col)
            
            self.classes = [str(c) for c in (model_info.get("classes") or [])]
            self.model_name = model_info.get("saved_model_name") or model_info.get("model_name")
            self.model_version = tag
            self.loaded_at = datetime.utcnow()

            logger.info("Model loaded successfully", tag=tag, cats=len(self.categorical_cols), nums=len(self.numeric_cols))
            return True
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(str(e))

    def is_model_loaded(self) -> bool:
        return self.model is not None

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
            
            track_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
            return results, probs, track_id
            
        except Exception as e:
            logger.error("Prediction failed Detailed", error=str(e))
            raise PredictionError(f"Prediction failed: {str(e)}")

model_service = ModelService()
