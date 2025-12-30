"""
Logistic Regression Model
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.model_training.models.base_model import BaseModel
from src.common.logger import logger


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""

    def __init__(self, config: dict):
        super().__init__("logistic_regression", config)
        self.params = config.get("logistic_regression", {}).get("params", {})

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        logger.info(f"\n🚀 Training {self.name.upper()}...")

        self.feature_names = X_train.columns.tolist()

        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)

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

        if hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
            df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importance,
                }
            ).sort_values("importance", ascending=False)
            return df

        return None