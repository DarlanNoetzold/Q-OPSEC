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
            logger.debug(f"Models root {self.models_root} does not exist.")
            return None
        dirs = [d for d in self.models_root.iterdir() if d.is_dir()]
        if not dirs:
            logger.debug(f"No version directories found under {self.models_root}")
            return None
        latest = sorted(dirs)[-1]
        logger.debug(f"Latest version directory resolved to: {latest}")
        return latest

    def _normalize_feature_names(self, raw) -> List[str]:
        """
        Normalize possible formats of feature_names.json into a plain list[str].
        Supported inputs:
         - list[str]
         - dict with 'all_features' key
         - dict with numeric-string keys like {"0": "f1", "1": "f2"}
         - dict inverted mapping {"f1": 0, "f2": 1}
         - fallback -> first list inside dict or dict.keys()
        """
        logger.debug(f"Normalizing raw feature_names of type {type(raw)}")
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            # explicit key
            if "all_features" in raw and isinstance(raw["all_features"], list):
                return raw["all_features"]
            # other named lists
            for candidate in ("features", "feature_list", "numeric_features", "categorical_features"):
                if candidate in raw and isinstance(raw[candidate], list):
                    return raw[candidate]
            # numeric-string keys -> sort by int(key)
            try:
                items = [(int(k), v) for k, v in raw.items()]
                items_sorted = [v for _, v in sorted(items)]
                logger.debug("Feature names parsed from numeric-string keys.")
                return items_sorted
            except Exception:
                pass
            # inverted mapping value->index
            try:
                inverted = {int(v): k for k, v in raw.items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
                if inverted:
                    return [inverted[i] for i in sorted(inverted.keys())]
            except Exception:
                pass
            # find first list value
            for v in raw.values():
                if isinstance(v, list):
                    logger.debug("Feature names taken from first list value inside dict.")
                    return v
            # fallback: keys
            logger.debug("Falling back to dict keys as feature names.")
            return list(raw.keys())
        # fallback convert to str
        logger.warning("feature_names.json has unexpected structure; coercing to single-element list.")
        return [str(raw)]

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
            try:
                with open(feature_path, "r") as f:
                    raw = json.load(f)
                self.feature_names = self._normalize_feature_names(raw)
            except Exception as e:
                logger.exception(f"Failed to parse feature_names.json: {e}")
                self.feature_names = []
        else:
            logger.warning("feature_names.json not found in model folder")
            self.feature_names = []

        # Ensure feature_names is explicitly a list
        if not isinstance(self.feature_names, list):
            try:
                self.feature_names = list(self.feature_names or [])
            except Exception:
                self.feature_names = []
        logger.info(f"Loaded {len(self.feature_names)} feature_names (sample): {self.feature_names[:50]}")

        # load label encoders
        encoders_path = version_dir / "label_encoders.pkl"
        if encoders_path.exists():
            try:
                self.label_encoders = joblib.load(encoders_path)
                logger.info("Label encoders loaded.")
            except Exception as e:
                logger.warning(f"Could not load label_encoders.pkl: {e}")
                self.label_encoders = {}
        else:
            logger.info("label_encoders.pkl not found in model folder")

        # load scaler
        scaler_path = version_dir / "scaler.pkl"
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded.")
            except Exception as e:
                logger.warning(f"Could not load scaler.pkl: {e}")
                self.scaler = None
        else:
            logger.info("scaler.pkl not found in model folder")

        # load models and metadata
        self.models = {}
        for p in version_dir.glob("*_model.pkl"):
            model_name = p.name.replace("_model.pkl", "")
            try:
                model = joblib.load(p)
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.exception(f"Failed to load model {p}: {e}")

        self.metadata = {}
        for m in version_dir.glob("*_metadata.json"):
            model_name = m.name.replace("_metadata.json", "")
            try:
                with open(m, "r") as f:
                    self.metadata[model_name] = json.load(f)
                logger.info(f"Loaded metadata for model: {model_name}")
            except Exception as e:
                logger.exception(f"Failed to load metadata {m}: {e}")

        return self.loaded_version

    def list_versions(self) -> List[str]:
        if not self.models_root.exists():
            return []
        return sorted([d.name for d in self.models_root.iterdir() if d.is_dir()])

    def _prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare pandas DataFrame from input records.
        Accepts:
         - list of flat dicts
         - list of {"features": {...}} envelopes
        Ensures the DataFrame contains all columns present in self.feature_names (adds NaN columns if missing).
        """
        logger.debug(f"_prepare_dataframe called with {len(records)} records.")
        normalized = []
        for i, r in enumerate(records):
            if isinstance(r, dict) and "features" in r and isinstance(r["features"], dict):
                normalized.append(r["features"])
            else:
                normalized.append(r)

        df = pd.DataFrame(normalized)
        logger.debug(f"Initial DataFrame shape: {df.shape}; columns: {df.columns.tolist()[:50]}")

        # Defensive: ensure feature_names is list
        if not isinstance(self.feature_names, list):
            logger.debug(f"feature_names is not list; coercing. type={type(self.feature_names)}")
            try:
                if isinstance(self.feature_names, dict) and "all_features" in self.feature_names:
                    self.feature_names = self.feature_names["all_features"]
                else:
                    self.feature_names = list(self.feature_names or [])
            except Exception:
                logger.exception("Failed to coerce feature_names to list; setting to []")
                self.feature_names = []

        # If we have expected feature ordering, ensure all exist and reorder
        if self.feature_names:
            logger.debug(f"Ordering DataFrame by feature_names (n={len(self.feature_names)}).")
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan
            try:
                df = df[self.feature_names]
            except Exception as e:
                # Provide rich debug info
                logger.error("Indexing df with feature_names failed.")
                logger.error(f"feature_names (type={type(self.feature_names)}): sample={self.feature_names[:60]}")
                logger.error(f"df.columns (sample)={df.columns.tolist()[:60]}")
                raise RuntimeError(f"Failed to index dataframe by feature_names: {e}")
        else:
            logger.debug("No feature_names provided; returning DataFrame as-is.")

        return df

    def _apply_label_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.label_encoders:
            return df
        for col, encoder in (self.label_encoders or {}).items():
            if col not in df.columns:
                continue
            try:
                vals = df[col].astype(str).fillna("__NA__").values
                transformed = encoder.transform(vals)
                df[col] = transformed
            except Exception:
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

    def _preprocess_lightgbm(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing for LightGBM")
        df = df.replace('', np.nan)
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                vals = df[col].astype(str).fillna("__NA__").values
                try:
                    transformed = encoder.transform(vals)
                    df[col] = transformed
                except Exception:
                    mapping = {c: i for i, c in enumerate(encoder.classes_)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)
        for col in self.label_encoders.keys():
            if col in df.columns:
                categories = list(self.label_encoders[col].classes_)
                fill_value = 'missing' if 'missing' in categories else (categories[0] if categories else None)
                df[col] = df[col].astype(object).fillna(fill_value)
                df[col] = pd.Categorical(df[col], categories=categories)
        for col in self.label_encoders.keys():
            if col in df.columns and df[col].isnull().any():
                categories = df[col].cat.categories
                fill_value = categories[0] if len(categories) > 0 else None
                df[col] = df[col].fillna(fill_value)
        for col in df.columns:
            if col not in self.label_encoders:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        return df

    def _preprocess_logistic_regression(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing for Logistic Regression")
        df = df.replace('', np.nan)
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                vals = df[col].astype(str).fillna("__NA__").values
                try:
                    transformed = encoder.transform(vals)
                    df[col] = transformed
                except Exception:
                    mapping = {c: i for i, c in enumerate(encoder.classes_)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)
        for col in self.label_encoders.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
        for col in df.columns:
            if col not in self.label_encoders:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        return df

    def _preprocess_random_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing for Random Forest")
        df = df.replace('', np.nan)
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                vals = df[col].astype(str).fillna("__NA__").values
                try:
                    transformed = encoder.transform(vals)
                    df[col] = transformed
                except Exception:
                    mapping = {c: i for i, c in enumerate(encoder.classes_)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)
        for col in self.label_encoders.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
        for col in df.columns:
            if col not in self.label_encoders:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        return df

    def _preprocess_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing for XGBoost")
        df = df.replace('', np.nan)
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                vals = df[col].astype(str).fillna("__NA__").values
                try:
                    transformed = encoder.transform(vals)
                    df[col] = transformed
                except Exception:
                    mapping = {c: i for i, c in enumerate(encoder.classes_)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)
        for col in self.label_encoders.keys():
            if col in df.columns:
                categories = list(self.label_encoders[col].classes_)
                fill_value = 'missing' if 'missing' in categories else (categories[0] if categories else None)
                df[col] = df[col].astype(object).fillna(fill_value)
                df[col] = pd.Categorical(df[col], categories=categories)
        for col in self.label_encoders.keys():
            if col in df.columns and df[col].isnull().any():
                categories = df[col].cat.categories
                fill_value = categories[0] if len(categories) > 0 else None
                df[col] = df[col].fillna(fill_value)
        for col in df.columns:
            if col not in self.label_encoders:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        return df

    def predict(self, records: List[Dict[str, Any]], model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.loaded_version:
            self.load()

        df = self._prepare_dataframe(records)
        logger.debug(f"Prepared DataFrame for prediction: shape={df.shape}; columns={df.columns.tolist()[:50]}")

        if df.shape[0] == 0:
            logger.warning("Empty input DataFrame to predict(); returning empty results.")
            return {"version": self.loaded_version, "n_records": 0, "models": {}}

        results = {"version": self.loaded_version, "n_records": len(df), "models": {}}
        target_models = model_names or list(self.models.keys())

        for m_name in target_models:
            model = self.models.get(m_name)
            if model is None:
                results["models"][m_name] = {"error": "model not loaded"}
                continue

            try:
                # Pré-processamento específico por modelo
                if m_name == "lightgbm":
                    df_preprocessed = self._preprocess_lightgbm(df.copy())
                elif m_name == "logistic_regression":
                    df_preprocessed = self._preprocess_logistic_regression(df.copy())
                elif m_name == "random_forest":
                    df_preprocessed = self._preprocess_random_forest(df.copy())
                elif m_name == "xgboost":
                    df_preprocessed = self._preprocess_xgboost(df.copy())
                else:
                    # Default: aplicar label encoders e scaler
                    df_preprocessed = self._apply_label_encoders(df.copy())
                    df_preprocessed = self._apply_scaler(df_preprocessed)

                # Aplicar scaler se disponível e se não for logistic regression (que pode não precisar)
                if m_name != "logistic_regression":
                    df_scaled = self._apply_scaler(df_preprocessed.copy())
                else:
                    df_scaled = df_preprocessed

                pos = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df_scaled)
                    if getattr(proba, "ndim", 1) > 1 and proba.shape[1] > 1:
                        pos = proba[:, 1].astype(float).tolist()
                    else:
                        pos = pd.Series(proba.ravel()).astype(float).tolist()
                else:
                    if hasattr(model, "decision_function"):
                        scores = model.decision_function(df_scaled)
                        pos = (1 / (1 + np.exp(-scores))).astype(float).tolist()
                    else:
                        preds = model.predict(df_scaled)
                        pos = pd.Series(preds).astype(float).tolist()

                threshold = float(self.metadata.get(m_name, {}).get("optimal_threshold", 0.5))
                preds = [(1 if p >= threshold else 0) for p in pos]

                results["models"][m_name] = {
                    "probabilities": pos,
                    "predictions": preds,
                    "threshold": threshold,
                    "metadata": self.metadata.get(m_name, {}),
                }
            except Exception as e:
                logger.exception(f"Prediction failed for model {m_name}: {e}")
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