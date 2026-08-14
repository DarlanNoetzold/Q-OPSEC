
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json

from src.common.logger import logger


class BaseModel(ABC):
    

    def __init__(self, config: dict, model_name: str):
        
        self.config = config
        self.model_name = model_name
        self.name = model_name
        self.model = None
        self.best_threshold = 0.5
        self.feature_names = None
        self.feature_importance = None

        logger.info(f"Initialized {self.model_name} model")

    @abstractmethod
    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> None:
        
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        
        pass

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        
        if threshold is None:
            threshold = self.best_threshold

        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        
        if self.feature_importance is None:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def save(self, output_dir: Path, version: str) -> None:
        
        model_dir = output_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{self.model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"   💾 Model saved to {model_path}")

        metadata = {
            'model_name': self.model_name,
            'best_threshold': float(self.best_threshold),
            'feature_names': self.feature_names,
            'config': self.config
        }

        metadata_path = model_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"   💾 Metadata saved to {metadata_path}")

        if self.feature_importance is not None:
            importance_df = self.get_feature_importance()
            importance_path = model_dir / f"{self.model_name}_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"   💾 Feature importance saved to {importance_path}")

    def load(self, model_dir: Path) -> None:
        
        model_path = model_dir / f"{self.model_name}_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"   📂 Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        metadata_path = model_dir / f"{self.model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.best_threshold = metadata.get('best_threshold', 0.5)
            self.feature_names = metadata.get('feature_names')
            logger.info(f"   📂 Metadata loaded from {metadata_path}")

    def optimize_threshold(
            self,
            y_true: np.ndarray,
            y_proba: np.ndarray,
            metric: str = 'f1',
            search_range: Tuple[float, float] = (0.1, 0.9),
            search_steps: int = 81
    ) -> float:
        
        from sklearn.metrics import precision_score, recall_score, f1_score

        thresholds = np.linspace(search_range[0], search_range[1], search_steps)
        best_score = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold
        logger.info(f"   🎯 Optimal threshold: {best_threshold:.4f} ({metric}={best_score:.4f})")

        return best_threshold

    def __str__(self) -> str:
        
        return f"{self.model_name.upper()} Model"

    def __repr__(self) -> str:
        
        return self.__str__()