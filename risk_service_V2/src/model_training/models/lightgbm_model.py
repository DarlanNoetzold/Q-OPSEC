"""
LightGBM Model
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

from src.model_training.models.base_model import BaseModel
from src.common.logger import logger


class LightGBMModel(BaseModel):
    """LightGBM classifier."""

    def __init__(self, config: dict):
        super().__init__(config, model_name='lightgbm')

        self.model_config = config.get('models', {}).get('lightgbm', {})
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
        early_stopping_rounds = early_stopping_config.get("rounds", 30)

        callbacks = []
        if early_stopping_enabled:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            )

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks if callbacks else None,
        )

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

        return df
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.name} is not trained yet")
        return self.model.score(X, y)