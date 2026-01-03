import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.common.logger import logger


class ModelManager:
    """Load trained models, artifacts and provide prediction utilities."""

    def __init__(self, models_root: str = "output/models", eval_root: str = "output/evaluation"):
        self.models_root = Path(models_root)
        self.eval_root = Path(eval_root)
        self.loaded_version: Optional[str] = None
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.feature_names: List[str] = []
        self.label_encoders = {}
        self.scaler = None

    def _get_latest_version_dir(self) -> Optional[Path]:
        if not self.models_root.exists():
            return None
        dirs = [d for d in self.models_root.iterdir() if d.is_dir()]
        if not dirs:
            return None
        # choose by folder name lexicographically (vYYYYMMDD_hhmmss) - should work
        latest = sorted(dirs)[-1]
        return latest

    def load(self, version: Optional[str] = None) -> str:
        """Load models and artifacts for a given version (or latest if None). Returns loaded version."""
        if version is None:
            version_dir = self._get_latest_version_dir()
            if version_dir is None:
                raise FileNotFoundError("No model versions found in output/models")
        else:
            version_dir = self.models_root / version
            if not version_dir.exists():
                raise FileNotFoundError(f"Version {version} not found in {self.models_root}")

        self.loaded_version = version_dir.name
        logger.info(f"Loading models from {version_dir}")

        # load feature names
        feature_path = version_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, "r") as f:
                self.feature_names = json.load(f)
        else:
            logger.warning("feature_names.json not found in model folder")

        # load label encoders
        encoders_path = version_dir / "label_encoders.pkl"
        if encoders_path.exists():
            self.label_encoders = joblib.load(encoders_path)
        else:
            logger.warning("label_encoders.pkl not found in model folder")

        # load scaler
        scaler_path = version_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            logger.warning("scaler.pkl not found in model folder")

        # load models and metadata
        for p in version_dir.glob("*_model.pkl"):
            model_name = p.name.replace("_model.pkl", "")
            try:
                model = joblib.load(p)
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {p}: {e}")

        for m in version_dir.glob("*_metadata.json"):
            model_name = m.name.replace("_metadata.json", "")
            try:
                with open(m, "r") as f:
                    self.metadata[model_name] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata {m}: {e}")

        return self.loaded_version

    def list_versions(self) -> List[str]:
        if not self.models_root.exists():
            return []
        return sorted([d.name for d in self.models_root.iterdir() if d.is_dir()])

    def _prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        # ensure all feature columns exist
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[self.feature_names]
        return df

    def _apply_label_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        # label_encoders expected to be dict feature->encoder (sklearn LabelEncoder or similar)
        for col, encoder in (self.label_encoders or {}).items():
            if col not in df.columns:
                continue
            try:
                vals = df[col].astype(str).fillna("__NA__").values
                transformed = encoder.transform(vals)
                df[col] = transformed
            except Exception:
                # fallback: try to map using classes_
                try:
                    mapping = {c: i for i, c in enumerate(encoder.classes_)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)
                except Exception:
                    df[col] = -1
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            return df
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}")
        return df

    def predict(self, records: List[Dict[str, Any]], model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict on a list of records. Returns dict with per-model probabilities and predictions."""
        if not self.loaded_version:
            self.load()

        df = self._prepare_dataframe(records)
        df_enc = self._apply_label_encoders(df.copy())
        df_scaled = self._apply_scaler(df_enc.copy())

        results = {"version": self.loaded_version, "n_records": len(df), "models": {}}

        target_models = model_names or list(self.models.keys())

        for m_name in target_models:
            model = self.models.get(m_name)
            if model is None:
                results["models"][m_name] = {"error": "model not loaded"}
                continue

            try:
                proba = model.predict_proba(df_scaled)
                if proba.ndim > 1:
                    pos = proba[:, 1].tolist()
                else:
                    pos = proba.tolist()

                threshold = self.metadata.get(m_name, {}).get("optimal_threshold", 0.5)
                preds = [(1 if p >= threshold else 0) for p in pos]

                results["models"][m_name] = {
                    "probabilities": pos,
                    "predictions": preds,
                    "threshold": threshold,
                    "metadata": self.metadata.get(m_name, {}),
                }
            except Exception as e:
                results["models"][m_name] = {"error": str(e)}

        return results

    def get_model_metrics(self, version: Optional[str] = None) -> Dict[str, Any]:
        if version is None:
            version = self.loaded_version or (self._get_latest_version_dir().name if self._get_latest_version_dir() else None)
        if version is None:
            return {}
        eval_path = self.eval_root / version / "metrics.json"
        if not eval_path.exists():
            logger.warning(f"Metrics file not found for version {version}")
            return {}
        with open(eval_path, "r") as f:
            return json.load(f)

    def get_dataset_summary(self) -> str:
        summary_path = Path("output/dataset_summary.txt")
        if summary_path.exists():
            return summary_path.read_text()
        return "No dataset summary available"