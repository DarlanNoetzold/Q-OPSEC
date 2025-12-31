"""
Model Factory - Creates configured model instances
"""
from typing import Any, Dict

from src.common.logger import logger

from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel


class ModelFactory:
    """
    Factory to create ML model instances based on config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: training config dict (usually loaded from YAML)
        """
        self.config = config
        self.models_config = config.get("models", {})

    def create_model(self, model_name: str):
        """
        Create a model instance by name using config.

        Args:
            model_name: name of the model in config (e.g., "logistic_regression")

        Returns:
            Instantiated model (sklearn / xgboost / lightgbm estimator)
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model '{model_name}' not found in config['models']")

        model_cfg = self.models_config[model_name]
        model_type = model_cfg.get("type", model_name)

        logger.info(f"   🧩 Creating model '{model_name}' (type='{model_type}')")

        # You can map by `type` or by `model_name`
        key = model_type.lower()

        if key in ("logistic_regression", "logreg", "lr"):
            return LogisticRegressionModel(model_cfg).build()

        if key in ("random_forest", "rf"):
            return RandomForestModel(model_cfg).build()

        if key in ("lightgbm", "lgbm", "lgb"):
            return LightGBMModel(model_cfg).build()

        if key in ("xgboost", "xgb"):
            return XGBoostModel(model_cfg).build()

        raise ValueError(f"Unknown model type: '{model_type}' for model '{model_name}'")