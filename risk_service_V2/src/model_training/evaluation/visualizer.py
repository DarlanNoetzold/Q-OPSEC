"""
Visualizer - Create evaluation plots and charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.common.logger import logger


class Visualizer:
    """Create evaluation visualizations."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.viz_config = config.get("evaluation", {}).get("visualizations", {})
        self.enabled = self.viz_config.get("enabled", True)
        self.plots = self.viz_config.get("plots", [])

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def create_all_plots(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        feature_importance: pd.DataFrame = None,
        version: str = "v1",
    ):
        """Create all configured plots."""
        if not self.enabled:
            logger.info("   ⚠️  Visualizations disabled")
            return

        plot_dir = self.output_dir / version / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n📈 Creating visualizations for {model_name}...")

        if "confusion_matrix" in self.plots:
            self.plot_confusion_matrix(y_true, y_pred, model_name, plot_dir)

        if "roc_curve" in self.plots:
            self.plot_roc_curve(y_true, y_proba, model_name, plot_dir)

        if "precision_recall_curve" in self.plots:
            self.plot_precision_recall_curve(y_true, y_proba, model_name, plot_dir)

        if "feature_importance" in self.plots and feature_importance is not None:
            self.plot_feature_importance(feature_importance, model_name, plot_dir)

        logger.info(f"   ✅ Visualizations saved to {plot_dir}")

    def plot_confusion_matrix(
        self, y_true, y_pred, model_name: str, output_dir: Path
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        filepath = output_dir / "confusion_matrix.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, model_name: str, output_dir: Path):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filepath = output_dir / "roc_curve.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_precision_recall_curve(
        self, y_true, y_proba, model_name: str, output_dir: Path
    ):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_proba[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filepath = output_dir / "precision_recall_curve.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(
        self, feature_importance: pd.DataFrame, model_name: str, output_dir: Path
    ):
        """Plot feature importance."""
        top_n = self.config.get("evaluation", {}).get("feature_importance", {}).get("top_n", 30)
        df = feature_importance.head(top_n)

        plt.figure(figsize=(10, max(8, len(df) * 0.3)))
        plt.barh(df["feature"], df["importance"], color="steelblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {len(df)} Feature Importance - {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        filepath = output_dir / "feature_importance.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_threshold_analysis(
        self,
        y_true,
        y_proba,
        model_name: str,
        output_dir: Path,
        thresholds=None,
    ):
        """Plot threshold analysis."""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)

        from sklearn.metrics import precision_score, recall_score, f1_score

        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label="Precision", marker="o", markersize=3)
        plt.plot(thresholds, recalls, label="Recall", marker="s", markersize=3)
        plt.plot(thresholds, f1_scores, label="F1 Score", marker="^", markersize=3)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Threshold Analysis - {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filepath = output_dir / "threshold_analysis.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()