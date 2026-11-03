"""
Model Training module.

Keep package imports lightweight: serialized inference models import modules from
this package, so training-only dependencies must not be imported at package load.
"""

__all__ = ["ModelTrainer", "DataLoader", "FeatureEngineer"]


def __getattr__(name):
    if name == "ModelTrainer":
        from src.model_training.trainer import ModelTrainer
        return ModelTrainer
    if name == "DataLoader":
        from src.model_training.data_loader import DataLoader
        return DataLoader
    if name == "FeatureEngineer":
        from src.model_training.feature_engineering import FeatureEngineer
        return FeatureEngineer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
