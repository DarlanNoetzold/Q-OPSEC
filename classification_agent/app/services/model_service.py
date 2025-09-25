"""
Model service for loading and managing ML models.
"""
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import ModelInfo, ModelMetrics

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class ModelService:
    """Service for managing ML model lifecycle."""

    def __init__(self, registry_dir: Optional[Union[str, Path]] = None):
        self.registry_dir = Path(registry_dir or settings.model_registry_dir)
        self.pipeline = None
        self.classes: List[str] = []
        self.model_name: str = ""
        self.model_dir: Optional[Path] = None
        self.required_columns: List[str] = []
        self.model_info: Optional[ModelInfo] = None
        self.loaded_at: Optional[datetime] = None
        self.last_check: Optional[datetime] = None
        self._metrics_cache: Dict[str, Any] = {}

        # Try to load model on initialization
        try:
            self.load_latest_model()
        except Exception as e:
            logger.warning("Failed to load model on initialization", error=str(e))

    def _get_latest_model_info(self) -> Dict[str, Any]:
        """Get information about the latest model from registry."""
        latest_path = self.registry_dir / "latest.json"
        if not latest_path.exists():
            raise ModelLoadError(f"No latest.json found in {self.registry_dir}")

        try:
            with open(latest_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ModelLoadError(f"Failed to read latest.json: {e}")

    def _load_model_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """Load model metadata from metrics.json."""
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            raise ModelLoadError(f"No metrics.json found in {model_dir}")

        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ModelLoadError(f"Failed to read metrics.json: {e}")

    def _load_pipeline(self, model_path: Path) -> Any:
        """Load the ML pipeline from pickle file."""
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            raise ModelLoadError(f"Failed to load model from {model_path}: {e}")

    def _infer_required_columns(self) -> List[str]:
        """Infer required input columns from the model pipeline."""
        try:
            if not hasattr(self.pipeline, "named_steps"):
                return []

            preprocessor = self.pipeline.named_steps.get("pre")
            if not preprocessor or not hasattr(preprocessor, "transformers"):
                return []

            columns = []
            for name, transformer, column_selector in preprocessor.transformers:
                if isinstance(column_selector, list):
                    columns.extend(column_selector)

            # Remove duplicates while preserving order
            seen = set()
            unique_columns = []
            for col in columns:
                if col not in seen:
                    seen.add(col)
                    unique_columns.append(col)

            return unique_columns
        except Exception as e:
            logger.warning("Failed to infer required columns", error=str(e))
            return []

    def _should_reload_model(self) -> bool:
        """Check if model should be reloaded based on cache TTL and file changes."""
        if not settings.auto_reload_model:
            return False

        if not self.last_check:
            return True

        # Check cache TTL
        if datetime.utcnow() - self.last_check > timedelta(seconds=settings.model_cache_ttl):
            return True

        # Check if latest.json has been modified
        latest_path = self.registry_dir / "latest.json"
        if latest_path.exists():
            latest_mtime = datetime.fromtimestamp(latest_path.stat().st_mtime)
            if self.loaded_at and latest_mtime > self.loaded_at:
                return True

        return False

    def load_latest_model(self, force: bool = False) -> bool:
        """Load the latest model from registry."""
        if not force and not self._should_reload_model():
            return False

        try:
            # Get latest model info
            latest_info = self._get_latest_model_info()
            model_dir = Path(latest_info["dir"])

            # Check if this is the same model already loaded
            if not force and self.model_dir == model_dir:
                self.last_check = datetime.utcnow()
                return False

            # Load model metadata
            metadata = self._load_model_metadata(model_dir)

            # Load pipeline
            model_path = model_dir / "model.pkl"
            pipeline = self._load_pipeline(model_path)

            # Update instance state
            previous_model = self.model_name
            self.pipeline = pipeline
            self.classes = metadata.get("classes", [])
            self.model_name = metadata.get("saved_model_name", "unknown")
            self.model_dir = model_dir
            self.required_columns = self._infer_required_columns()
            self.loaded_at = datetime.utcnow()
            self.last_check = self.loaded_at

            # Create model info
            self.model_info = ModelInfo(
                saved_model_name=self.model_name,
                artifact_path=str(model_path.resolve()),
                classes=self.classes,
                required_columns=self.required_columns,
                cv_metrics=self._parse_metrics(metadata.get("cv_metrics", {})),
                holdout_metrics=self._parse_metrics(metadata.get("holdout_metrics", {})),
                meta=metadata.get("meta", {}),
                loaded_at=self.loaded_at,
                registry_dir=str(self.registry_dir.resolve()),
                model_dir=str(model_dir.resolve())
            )

            logger.info(
                "Model loaded successfully",
                model_name=self.model_name,
                previous_model=previous_model,
                classes_count=len(self.classes),
                required_columns_count=len(self.required_columns)
            )

            return True

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(f"Failed to load model: {e}")

    def _parse_metrics(self, metrics_dict: Dict[str, Any]) -> Optional[ModelMetrics]:
        """Parse metrics dictionary into ModelMetrics object."""
        if not metrics_dict:
            return None

        return ModelMetrics(
            accuracy=metrics_dict.get("accuracy"),
            f1_macro=metrics_dict.get("f1_macro"),
            precision=metrics_dict.get("precision"),
            recall=metrics_dict.get("recall")
        )

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.pipeline is not None

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the currently loaded model."""
        if not self.is_model_loaded():
            return None
        return self.model_info

    def _prepare_input_dataframe(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Prepare input data as DataFrame with proper column handling."""
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # Validate required columns
        if self.required_columns:
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                # Add missing columns with None values
                for col in missing_columns:
                    df[col] = None
                logger.warning(
                    "Missing input columns filled with None",
                    missing_columns=missing_columns
                )

        return df

    def _calculate_input_hash(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """Calculate hash of input data for tracking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def predict(
            self,
            data: Union[Dict[str, Any], List[Dict[str, Any]]],
            include_probabilities: bool = True
    ) -> Tuple[List[str], Optional[List[Dict[str, float]]], List[str]]:
        """
        Make predictions on input data.

        Returns:
            Tuple of (labels, probabilities, input_hashes)
        """
        if not self.is_model_loaded():
            raise PredictionError("No model loaded")

        try:
            # Prepare input
            df = self._prepare_input_dataframe(data)

            # Calculate input hashes
            if isinstance(data, dict):
                input_hashes = [self._calculate_input_hash(data)]
            else:
                input_hashes = [self._calculate_input_hash(record) for record in data]

            # Make predictions
            y_pred_indices = self.pipeline.predict(df)
            if not isinstance(y_pred_indices, (list, tuple, np.ndarray)):
                y_pred_indices = [y_pred_indices]

            # Convert indices to labels
            labels = [self.classes[int(idx)] for idx in y_pred_indices]

            # Get probabilities if requested and available
            probabilities = None
            if include_probabilities and hasattr(self.pipeline, "predict_proba"):
                try:
                    proba_matrix = self.pipeline.predict_proba(df)
                    probabilities = []
                    for i, proba_row in enumerate(proba_matrix):
                        prob_dict = {
                            self.classes[j]: float(proba_row[j])
                            for j in range(len(self.classes))
                        }
                        probabilities.append(prob_dict)
                except Exception as e:
                    logger.warning("Failed to get probabilities", error=str(e))
                    probabilities = None

            return labels, probabilities, input_hashes

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise PredictionError(f"Prediction failed: {e}")

    def validate_input(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[str]:
        """Validate input data and return list of validation errors."""
        errors = []

        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data

        for i, record in enumerate(data_list):
            if not isinstance(record, dict):
                errors.append(f"Record {i}: must be a dictionary")
                continue

            if not record:
                errors.append(f"Record {i}: cannot be empty")
                continue

            # Check for required columns (if we have them)
            if self.required_columns:
                missing = [col for col in self.required_columns if col not in record]
                if missing:
                    errors.append(f"Record {i}: missing required columns: {missing}")

        return errors


# Global model service instance
model_service = ModelService()