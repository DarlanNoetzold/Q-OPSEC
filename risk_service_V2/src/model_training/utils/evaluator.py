"""
Model Evaluator - Evaluate model performance
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.common.logger import logger


class ModelEvaluator:
    """Evaluate model performance."""

    def __init__(self, config: dict):
        """
        Initialize ModelEvaluator.

        Args:
            config: Training configuration dictionary
        """
        self.config = config

    def evaluate_model(
            self,
            model,
            X: pd.DataFrame,
            y_true: pd.Series,
            model_name: str,
            threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            model_name: Name of the model
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # 👇 CORREÇÃO: Garantir que y_true é 1D array sem NaN
            if isinstance(y_true, pd.DataFrame):
                y_true = y_true.values.ravel()
            elif isinstance(y_true, pd.Series):
                y_true = y_true.values

            # Remove NaN do y_true e correspondentes em X
            mask = ~pd.isna(y_true)
            if not mask.all():
                logger.warning(f"   ⚠️  Removing {(~mask).sum()} NaN values from y_true for {model_name}")
                y_true = y_true[mask]
                X = X[mask]

            # Ensure y_true is integer
            y_true = y_true.astype(int)

            # Get predictions
            y_proba = model.predict_proba(X)

            # 👇 CORREÇÃO: Garantir que y_proba é 1D array
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]  # Probabilidade da classe 1

            y_pred = (y_proba >= threshold).astype(int)

            # Calculate metrics
            metrics = {
                "model_name": model_name,
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_proba)
            }

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }

            # Log metrics
            logger.info(f"\n📊 {model_name.upper()} Evaluation:")
            logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall:    {metrics['recall']:.4f}")
            logger.info(f"   F1 Score:  {metrics['f1']:.4f}")
            logger.info(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
            logger.info(f"   Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            raise

    def compare_models(self, all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            all_metrics: Dictionary of model metrics

        Returns:
            DataFrame with comparison
        """
        if not all_metrics:
            logger.warning("⚠️  No metrics to compare")
            return pd.DataFrame()

        comparison_data = []
        for model_name, metrics in all_metrics.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0),
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1": metrics.get("f1", 0),
                "ROC AUC": metrics.get("roc_auc", 0)
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("F1", ascending=False)

        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        logger.info("\n" + df.to_string(index=False))

        return df