"""
Trainer - Main training orchestrator
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime
import json

from src.common.logger import logger
from src.model_training.data_loader import DataLoader
from src.model_training.feature_engineering import FeatureEngineer
from src.model_training.utils.model_registry import ModelRegistry
from src.model_training.evaluation.metrics import MetricsCalculator
from src.model_training.evaluation.visualizer import Visualizer


class ModelTrainer:
    """Orchestrate the complete model training pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.output_config = config.get("output", {})
        self.training_config = config.get("training", {})

        self.models_dir = Path(self.output_config.get("models_dir", "output/models"))
        self.eval_dir = Path(self.output_config.get("evaluation_dir", "output/evaluation"))

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.version = self._generate_version()

        self.data_loader = DataLoader(config)
        self.model_registry = ModelRegistry(config)
        self.metrics_calculator = MetricsCalculator(config)
        self.visualizer = Visualizer(config, self.eval_dir)

        self.models = {}
        self.all_metrics = {}

        logger.info("=" * 80)
        logger.info("MODEL TRAINER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Version: {self.version}")
        logger.info(f"Models dir: {self.models_dir}")
        logger.info(f"Evaluation dir: {self.eval_dir}")

    def _generate_version(self) -> str:
        """Generate version string."""
        versioning = self.output_config.get("versioning", {})
        if not versioning.get("enabled", True):
            return "v1"

        format_type = versioning.get("format", "timestamp")
        if format_type == "timestamp":
            return datetime.now().strftime("v%Y%m%d_%H%M%S")
        else:
            return "v1"

    def run(self):
        """Run the complete training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        # 1. Load data
        train_df, val_df, test_df = self.data_loader.load_datasets()

        # 2. Prepare features
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_features(
            train_df, val_df, test_df
        )

        # 3. Feature engineering
        feature_info = self.data_loader.get_feature_info()
        feature_engineer = FeatureEngineer(self.config, feature_info)
        X_train, X_val, X_test = feature_engineer.fit_transform(
            X_train, y_train, X_val, X_test
        )

        # 4. Initialize models
        self.models = self.model_registry.create_models()

        if not self.models:
            logger.error("❌ No models to train. Exiting.")
            return

        # 5. Train models
        self._train_all_models(X_train, y_train, X_val, y_val)

        # 6. Optimize thresholds
        if self.training_config.get("threshold_optimization", {}).get("enabled", True):
            self._optimize_thresholds(X_val, y_val)

        # 7. Evaluate on test set
        self._evaluate_all_models(X_test, y_test)

        # 8. Save models and artifacts
        self._save_all_artifacts(feature_engineer, X_train.columns.tolist())

        # 9. Generate comparison report
        self._generate_comparison_report()

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Version: {self.version}")
        logger.info(f"Models trained: {len(self.models)}")
        logger.info(f"Artifacts saved to: {self.models_dir / self.version}")

    def _train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all models."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING MODELS")
        logger.info("=" * 80)

        for model_name, model in self.models.items():
            try:
                model.train(X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                del self.models[model_name]

    def _optimize_thresholds(self, X_val, y_val):
        """Optimize classification thresholds."""
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZING THRESHOLDS")
        logger.info("=" * 80)

        threshold_config = self.training_config.get("threshold_optimization", {})
        metric = threshold_config.get("metric", "f1")
        search_range = threshold_config.get("search_range", [0.1, 0.9])
        search_steps = threshold_config.get("search_steps", 81)

        for model_name, model in self.models.items():
            try:
                model.optimize_threshold(
                    X_val, y_val, metric, tuple(search_range), search_steps
                )
            except Exception as e:
                logger.warning(f"⚠️  Threshold optimization failed for {model_name}: {e}")

    def _evaluate_all_models(self, X_test, y_test):
        """Evaluate all models on test set."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING MODELS ON TEST SET")
        logger.info("=" * 80)

        for model_name, model in self.models.items():
            try:
                y_proba = model.predict_proba(X_test)
                y_pred = model.predict(X_test)

                metrics = self.metrics_calculator.calculate_metrics(
                    y_test, y_pred, y_proba, "test"
                )
                self.all_metrics[model_name] = metrics

                self.metrics_calculator.log_metrics(metrics, model_name, "test")

                feature_importance = model.get_feature_importance()

                self.visualizer.create_all_plots(
                    model_name,
                    y_test,
                    y_pred,
                    y_proba,
                    feature_importance,
                    self.version,
                )

                if "threshold_analysis" in self.visualizer.plots:
                    plot_dir = self.eval_dir / self.version / model_name
                    self.visualizer.plot_threshold_analysis(
                        y_test, y_proba, model_name, plot_dir
                    )

            except Exception as e:
                logger.error(f"❌ Evaluation failed for {model_name}: {e}")

    def _save_all_artifacts(self, feature_engineer, feature_names):
        """Save all models and artifacts."""
        logger.info("\n" + "=" * 80)
        logger.info("SAVING ARTIFACTS")
        logger.info("=" * 80)

        save_config = self.output_config.get("save_artifacts", {})

        # Save models
        if save_config.get("model", True):
            for model_name, model in self.models.items():
                try:
                    model.save(self.models_dir, self.version)
                except Exception as e:
                    logger.error(f"❌ Failed to save {model_name}: {e}")

        # Save preprocessor
        if save_config.get("preprocessor", True):
            try:
                feature_engineer.save_preprocessor(self.models_dir, self.version)
            except Exception as e:
                logger.error(f"❌ Failed to save preprocessor: {e}")

        # Save feature names
        if save_config.get("feature_names", True):
            feature_path = self.models_dir / self.version / "feature_names.json"
            with open(feature_path, "w") as f:
                json.dump(feature_names, f, indent=2)
            logger.info(f"   💾 Feature names saved to {feature_path}")

        # Save metrics
        if save_config.get("metrics", True):
            metrics_path = self.eval_dir / self.version / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(self.all_metrics, f, indent=2)
            logger.info(f"   💾 Metrics saved to {metrics_path}")

        # Save config
        if save_config.get("config", True):
            config_path = self.models_dir / self.version / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"   💾 Config saved to {config_path}")

    def _generate_comparison_report(self):
        """Generate model comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)

        if not self.all_metrics:
            logger.warning("⚠️  No metrics to compare")
            return

        comparison_df = self.metrics_calculator.compare_models(self.all_metrics)

        logger.info("\n" + comparison_df.to_string())

        report_path = self.eval_dir / self.version / "model_comparison.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(report_path)
        logger.info(f"\n💾 Comparison report saved to {report_path}")

        best_model = comparison_df.index[0]
        best_auc = comparison_df.loc[best_model, "roc_auc"]
        logger.info(f"\n🏆 Best model: {best_model} (ROC-AUC: {best_auc:.4f})")