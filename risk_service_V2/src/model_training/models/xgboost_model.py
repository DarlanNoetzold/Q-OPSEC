"""
XGBoost Model
"""
import pandas as pd
import numpy as np
import xgboost as xgb

from src.model_training.models.base_model import BaseModel
from src.common.logger import logger


class XGBoostModel(BaseModel):
    """XGBoost classifier."""

    def __init__(self, config: dict):
        super().__init__(config, model_name='xgboost')

        self.model_config = config.get('models', {}).get('xgboost', {})
        self.params = self.model_config.get('params', {})

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        logger.info(f"\n🚀 Training {self.name.upper()}...")

        self.feature_names = X_train.columns.tolist()

        early_stopping_config = self.config.get("training", {}).get("early_stopping", {})
        early_stopping_enabled = early_stopping_config.get("enabled", True)

        eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model = xgb.XGBClassifier(**self.params)

        if early_stopping_enabled:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        self.is_trained = True

        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        logger.info(f"   ✅ {self.name} trained")
        logger.info(f"      Train accuracy: {train_score:.4f}")
        logger.info(f"      Val accuracy:   {val_score:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.name} is not trained yet")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame | None:
        if not self.is_trained or self.model is None:
            return None

        importance = self.model.feature_importances_
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.name} is not trained yet")
        return self.model.score(X, y)

        return df