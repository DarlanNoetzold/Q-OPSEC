
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import joblib
from datetime import datetime

from src.common.logger import logger
from src.model_training.data_loader import DataLoader
from src.model_training.feature_engineering import FeatureEngineer
from src.model_training.models.model_factory import ModelFactory
from src.model_training.utils.evaluator import ModelEvaluator


class ModelTrainer:
    

    def __init__(self, config: dict):
        
        self.config = config
        self.version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_factory = ModelFactory(config)
        self.evaluator = ModelEvaluator(config)

        self.models_dir = Path(config.get("output", {}).get("models_dir", "output/models"))
        self.eval_dir = Path(config.get("output", {}).get("evaluation_dir", "output/evaluation"))

        self.models_dir = self.models_dir / self.version
        self.eval_dir = self.eval_dir / self.version

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.optimal_thresholds: Dict[str, float] = {}

        logger.info("=" * 80)
        logger.info("MODEL TRAINER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Version: {self.version}")
        logger.info(f"Models dir: {self.models_dir}")
        logger.info(f"Evaluation dir: {self.eval_dir}")

    def train_pipeline(self) -> None:
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        train_df, val_df, test_df = self.data_loader.load_datasets()

        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_features(
            train_df, val_df, test_df
        )

        y_train = self._ensure_1d(y_train, "y_train")
        y_val = self._ensure_1d(y_val, "y_val")
        y_test = self._ensure_1d(y_test, "y_test")

        X_train, X_val, X_test = self.feature_engineer.fit_transform(
            X_train, X_val, X_test
        )

        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        for col in categorical_cols:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        self._initialize_models()

        self._train_models(X_train, y_train, X_val, y_val)

        self._optimize_thresholds(X_val, y_val)

        self._evaluate_models(X_test, y_test)

        self._save_artifacts()

        self._compare_models()

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Version: {self.version}")
        logger.info(f"Models trained: {len(self.trained_models)}")
        logger.info(f"Artifacts saved to: {self.models_dir}")

    def _ensure_1d(self, y: pd.Series, name: str) -> np.ndarray:
        
        if isinstance(y, pd.DataFrame):
            logger.warning(f"   ⚠️  {name} is DataFrame, converting to 1D array")
            y = y.values.ravel()
        elif isinstance(y, pd.Series):
            y = y.values

        if len(y.shape) > 1:
            logger.warning(f"   ⚠️  {name} is {y.shape}, flattening to 1D")
            y = y.ravel()

        if np.isnan(y).any():
            logger.warning(f"   ⚠️  {name} contains {np.isnan(y).sum()} NaN values")
            raise ValueError(f"{name} contains NaN values after cleaning")

        y = y.astype(int)

        logger.info(f"   ✅ {name} shape: {y.shape}, dtype: {y.dtype}, unique: {np.unique(y)}")

        return y

    def _initialize_models(self) -> None:
        
        logger.info("\n" + "=" * 80)
        logger.info("INITIALIZING MODELS")
        logger.info("=" * 80)

        model_configs = self.config.get("models", {})

        enabled_models = model_configs.get("enabled", [])

        if not enabled_models:
            raise ValueError("No models enabled in configuration (models.enabled is empty)")

        logger.info(f"   Enabled models: {enabled_models}")

        for model_name in enabled_models:
            if model_name in model_configs:
                try:
                    model = self.model_factory.create_model(model_name)
                    self.trained_models[model_name] = {
                    "model": model,
                    "config": model_configs[model_name],
                    "trained": False
                    }
                    logger.info(f"   ✅ {model_name} initialized")
                except Exception as e:
                    logger.error(f"   ❌ Failed to initialize {model_name}: {e}")
            else:
                logger.warning(f"   ⚠️  {model_name} enabled but no config found, skipping")

        if not self.trained_models:
            raise ValueError("No models successfully initialized")

        logger.info(f"\n📊 Total models to train: {len(self.trained_models)}")

    def _train_models(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame,
            y_val: np.ndarray
    ) -> None:
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING MODELS")
        logger.info("=" * 80)

        y_train_orig = pd.Series(y_train)
        y_val_orig = pd.Series(y_val)

        for model_name, model_info in self.trained_models.items():
            logger.info(f"\n🚀 Training {model_name.upper()}...")

            try:
                model = model_info["model"]

                model.train(X_train, y_train_orig, X_val, y_val_orig)

                train_acc = model.score(X_train, y_train)
                val_acc = model.score(X_val, y_val)

                logger.info(f"   ✅ {model_name} trained")
                logger.info(f"      Train accuracy: {train_acc:.4f}")
                logger.info(f"      Val accuracy:   {val_acc:.4f}")

                model_info["trained"] = True
                model_info["train_accuracy"] = train_acc
                model_info["val_accuracy"] = val_acc

            except Exception as e:
                logger.error(f"   ❌ Training failed for {model_name}: {e}")
                model_info["trained"] = False

    def _optimize_thresholds(
            self,
            X_val: pd.DataFrame,
            y_val: np.ndarray
    ) -> None:
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZING THRESHOLDS")
        logger.info("=" * 80)

        from sklearn.metrics import f1_score

        for model_name, model_info in self.trained_models.items():
            if not model_info.get("trained", False):
                continue

            try:
                model = model_info["model"]

                y_proba = model.predict_proba(X_val)

                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]

                thresholds = np.arange(0.1, 0.9, 0.05)
                best_f1 = 0
                best_threshold = 0.5

                for threshold in thresholds:
                    y_pred = (y_proba >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred, zero_division=0)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

                self.optimal_thresholds[model_name] = best_threshold
                logger.info(f"   ✅ {model_name}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")

            except Exception as e:
                logger.warning(f"   ⚠️  Threshold optimization failed for {model_name}: {e}")
                self.optimal_thresholds[model_name] = 0.5

    def _evaluate_models(
            self,
            X_test: pd.DataFrame,
            y_test: np.ndarray
    ) -> None:
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING MODELS ON TEST SET")
        logger.info("=" * 80)

        for model_name, model_info in self.trained_models.items():
            if not model_info.get("trained", False):
                continue

            try:
                model = model_info["model"]
                threshold = self.optimal_thresholds.get(model_name, 0.5)

                metrics = self.evaluator.evaluate_model(
                    model=model,
                    X=X_test,
                    y_true=y_test,
                    model_name=model_name,
                    threshold=threshold
                )

                self.metrics[model_name] = metrics

            except Exception as e:
                logger.error(f"   ❌ Evaluation failed for {model_name}: {e}")

    def _save_artifacts(self) -> None:
        
        logger.info("\n" + "=" * 80)
        logger.info("SAVING ARTIFACTS")
        logger.info("=" * 80)

        for model_name, model_info in self.trained_models.items():
            if not model_info.get("trained", False):
                continue

            model_path = self.models_dir / f"{model_name}_model.pkl"
            joblib.dump(model_info["model"], model_path)
            logger.info(f"   💾 Model saved to {model_path}")

            metadata = {
                "model_name": model_name,
                "version": self.version,
                "train_accuracy": model_info.get("train_accuracy"),
                "val_accuracy": model_info.get("val_accuracy"),
                "optimal_threshold": self.optimal_thresholds.get(model_name, 0.5),
                "config": model_info.get("config", {})
            }

            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"   💾 Metadata saved to {metadata_path}")

        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(self.feature_engineer.scaler, scaler_path)
        logger.info(f"   💾 Scaler saved to {scaler_path}")

        encoders_path = self.models_dir / "label_encoders.pkl"
        joblib.dump(self.feature_engineer.label_encoders, encoders_path)
        logger.info(f"   💾 Label encoders saved to {encoders_path}")

        feature_names_path = self.models_dir / "feature_names.json"
        feature_info = self.data_loader.get_feature_info()
        with open(feature_names_path, "w") as f:
            json.dump(feature_info, f, indent=2)
        logger.info(f"   💾 Feature names saved to {feature_names_path}")

        metrics_path = self.eval_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"   💾 Metrics saved to {metrics_path}")

        config_path = self.models_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"   💾 Config saved to {config_path}")

    def _compare_models(self) -> None:
        
        comparison_df = self.evaluator.compare_models(self.metrics)

        if not comparison_df.empty:
            comparison_path = self.eval_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"\n💾 Comparison saved to {comparison_path}")