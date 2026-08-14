# model_factory.py
from src.model_training.models.xgboost_model import XGBoostModel
from src.model_training.models.lightgbm_model import LightGBMModel
from src.model_training.models.logistic_regression_model import LogisticRegressionModel
from src.model_training.models.random_forest_model import RandomForestModel

# Importar os novos modelos
from src.model_training.models.catboost_model import CatBoostModel
from src.model_training.models.pytorch_mlp_model import PyTorchMLPModel

class ModelFactory:
    def __init__(self, config):
        self.config = config

    def create_model(self, model_name: str):
        model_name = model_name.lower()
        if model_name == "xgboost":
            return XGBoostModel(self.config)
        elif model_name == "lightgbm":
            return LightGBMModel(self.config)
        elif model_name == "logistic_regression":
            return LogisticRegressionModel(self.config)
        elif model_name == "random_forest":
            return RandomForestModel(self.config)
        elif model_name == "catboost":
            return CatBoostModel(self.config)
        elif model_name == "pytorch_mlp":
            return PyTorchMLPModel(self.config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")