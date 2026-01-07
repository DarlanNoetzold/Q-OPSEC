# catboost_model.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from src.model_training.models.base_model import BaseModel
from src.common.logger import logger

class CatBoostModel(BaseModel):
    """CatBoost classifier with native categorical support."""

    def __init__(self, config: dict):
        super().__init__(config, model_name='catboost')
        self.model_config = config.get('models', {}).get('catboost', {})
        self.params = self.model_config.get('params', {})
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        logger.info(f"\n🚀 Training {self.name.upper()}...")

        cat_features = [i for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
            early_stopping_rounds=50,
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
        return self.model.predict_proba(X)[:, 1]

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.name} is not trained yet")
        return self.model.score(X, y)