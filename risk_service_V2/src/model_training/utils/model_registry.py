"""
Model Registry - Factory for creating model instances
"""
from typing import Dict, List
from src.model_training.models import (
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    LogisticRegressionModel,
)
from src.common.logger import logger


class ModelRegistry:
    """Factory for creating and managing model instances."""

    MODEL_CLASSES = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
    }

    def __init__(self, config: dict):
        self.config = config
        self.models_config = config.get("models", {})
        self.enabled_models = self.models_config.get("enabled", [])

    def create_models(self) -> Dict[str, object]:
        """Create all enabled model instances."""
        models = {}

        logger.info("\n" + "=" * 80)
        logger.info("INITIALIZING MODELS")
        logger.info("=" * 80)

        for model_name in self.enabled_models:
            if model_name not in self.MODEL_CLASSES:
                logger.warning(f"   ⚠️  Unknown model: {model_name}, skipping")
                continue

            try:
                model_class = self.MODEL_CLASSES[model_name]
                model_instance = model_class(self.config)
                models[model_name] = model_instance
                logger.info(f"   ✅ {model_name} initialized")
            except Exception as e:
                logger.error(f"   ❌ Failed to initialize {model_name}: {e}")

        logger.info(f"\n✅ {len(models)} models ready for training")
        return models

    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.MODEL_CLASSES.keys())