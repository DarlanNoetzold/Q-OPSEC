"""
Metrics - Calculate evaluation metrics for models
"""
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

from src.common.logger import logger


class MetricsCalculator:
    """Calculate and format evaluation metrics."""

    def __init__(self, config: dict):
        self.config = config
        self.metrics_config = config.get("training", {}).get("metrics", [])

    def calculate_metrics(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            y_proba: np.ndarray,
            dataset_name: str = "test",
    ) -> Dict[str, float]:
        """Calculate all configured metrics."""
        metrics = {}

        if "accuracy" in self.metrics_config:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

        if "precision" in self.metrics_config:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)

        if "recall" in self.metrics_config:
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

        if "f1" in self.metrics_config:
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        if "roc_auc" in self.metrics_config:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics["roc_auc"] = 0.0

        if "pr_auc" in self.metrics_config:
            try:
                metrics["pr_auc"] = average_precision_score(y_true, y_proba[:, 1])
            except:
                metrics["pr_auc"] = 0.0

        if "log_loss" in self.metrics_config:
            try:
                metrics["log_loss"] = log_loss(y_true, y_proba)
            except:
                metrics["log_loss"] = 0.0

        return metrics

    def log_metrics(self, metrics: Dict[str, float], model_name: str, dataset_name: str = "test"):
        """Log metrics in a formatted way."""
        logger.info(f"\n📊 {model_name.upper()} - {dataset_name.upper()} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"   {metric_name:12s}: {value:.4f}")

    def get_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
            self, y_true: pd.Series, y_pred: np.ndarray
    ) -> str:
        """Get classification report."""
        return classification_report(y_true, y_pred, zero_division=0)

    def compare_models(self, all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare metrics across models."""
        df = pd.DataFrame(all_metrics).T

        # Sort by roc_auc if available, otherwise by f1
        if "roc_auc" in df.columns:
            df = df.sort_values("roc_auc", ascending=False)
        elif "f1" in df.columns:
            df = df.sort_values("f1", ascending=False)

        return df