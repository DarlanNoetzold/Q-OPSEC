import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configs import DefaultConfig, ALLOWED_CLASSES
from registry import ModelRegistry
from trainer import train_and_select_best, TrainResult
from evaluator import evaluate_holdout
from scheduler import save_training_metrics, schedule_periodic_training


class ClassifySchedulerServiceTests(unittest.TestCase):
    def setUp(self):
        # Create a small deterministic dataset for service tests
        np.random.seed(42)
        n_samples = 40
        self.df = pd.DataFrame({
            "id": range(n_samples),
            "request_id_resolved": [f"req_{i}" for i in range(n_samples)],
            "created_at": ["2026-01-01T00:00:00"] * n_samples,
            "risk_score": np.random.uniform(0, 1, n_samples),
            "conf_score": np.random.uniform(0, 1, n_samples),
            "src_geo": np.random.choice(["US", "EU", "BR"], n_samples),
            "security_level_label": (["Critical", "High", "Medium", "Low"] * 10),
        })

    def test_train_and_select_best_with_small_models_subset(self):
        result = train_and_select_best(
            df=self.df,
            target_col="security_level_label",
            test_size=0.25,
            seed=42,
            n_splits=2,
            scoring_primary="accuracy",
            scoring_secondary="f1_macro",
            max_models=2,
        )

        self.assertIsInstance(result, TrainResult)
        self.assertIsNotNone(result.best_name)
        self.assertIn("accuracy", result.best_cv_metrics)
        self.assertIn("f1_macro", result.best_cv_metrics)
        self.assertGreaterEqual(result.best_cv_metrics["accuracy"], 0.0)
        self.assertGreater(len(result.candidates), 0)
        self.assertEqual(len(result.y_test), 10)  # 25% of 40 samples
        self.assertIsNotNone(result.pipeline)

    def test_train_and_select_best_fails_when_class_has_insufficient_samples(self):
        df_rare = pd.DataFrame({
            "risk_score": [0.1, 0.2, 0.3, 0.4],
            "security_level_label": ["Critical", "Critical", "Critical", "Low"]
        })
        with self.assertRaises(ValueError) as ctx:
            train_and_select_best(
                df=df_rare,
                target_col="security_level_label",
                test_size=0.25,
                seed=42,
                n_splits=2,
                scoring_primary="accuracy",
                scoring_secondary="f1_macro",
                max_models=1,
            )
        self.assertIn("Need at least 2 samples per class", str(ctx.exception))

    def test_evaluator_holdout_produces_valid_metrics(self):
        # Create a simple trained pipeline
        X = pd.DataFrame({"score": [0.1, 0.2, 0.8, 0.9]})
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        metrics = evaluate_holdout(pipe, X, y, allowed_classes=["Class0", "Class1"])
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)

    def test_model_registry_save_and_latest_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = ModelRegistry(Path(tmp_dir))
            pipe = Pipeline([("clf", LogisticRegression())])
            classes = ["Critical", "High"]
            cv_metrics = {"accuracy": 0.95, "f1_macro": 0.94}
            holdout_metrics = {"accuracy": 0.90, "f1_macro": 0.89}
            meta = {"seed": 42, "n_samples": 100}
            candidates = [{"name": "logreg_lbfgs", "cv_accuracy": 0.95, "cv_f1_macro": 0.94}]

            saved_path = registry.save(
                tag="20261025T120000Z",
                pipeline=pipe,
                classes=classes,
                cv_metrics=cv_metrics,
                holdout_metrics=holdout_metrics,
                model_name="test_model",
                meta=meta,
                candidates=candidates,
            )

            self.assertTrue(saved_path.is_dir())
            self.assertTrue((saved_path / "model.pkl").is_file())
            self.assertTrue((saved_path / "metrics.json").is_file())
            self.assertTrue((saved_path / "candidates.json").is_file())
            self.assertTrue((Path(tmp_dir) / "latest.json").is_file())

            # Verify unpickling
            with open(saved_path / "model.pkl", "rb") as f:
                loaded_pipe = pickle.load(f)
            self.assertIsInstance(loaded_pipe, Pipeline)

            # Verify latest.json structure
            with open(Path(tmp_dir) / "latest.json", "r") as f:
                latest_data = json.load(f)
            self.assertEqual("test_model", latest_data["saved_model_name"])
            self.assertEqual(classes, latest_data["classes"])
            self.assertEqual(str(saved_path.resolve()), latest_data["dir"])

    def test_save_training_metrics_generates_plots_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_dir = Path(tmp_dir) / "metrics"
            with patch("scheduler.METRICS_ROOT", metrics_dir):
                # Mock a TrainResult
                pipe = MagicMock()
                pipe.predict.return_value = np.array([0, 1, 0, 1])
                mock_result = TrainResult(
                    best_name="mock_best",
                    best_cv_metrics={"accuracy": 0.95, "f1_macro": 0.93},
                    pipeline=pipe,
                    label_encoder=MagicMock(),
                    X_test=pd.DataFrame({"a": [1, 2, 3, 4]}),
                    y_test=np.array([0, 1, 0, 1]),
                    candidates=[
                        {"name": "m1", "cv_accuracy": 0.85, "cv_f1_macro": 0.82},
                        {"name": "mock_best", "cv_accuracy": 0.95, "cv_f1_macro": 0.93},
                    ],
                    ordered_classes=["ClassA", "ClassB"],
                    feature_columns=["a"],
                )

                session_id = "test_session_123"
                save_training_metrics(mock_result, session_id)

                session_path = metrics_dir / session_id
                self.assertTrue(session_path.is_dir())
                self.assertTrue((session_path / "all_models_accuracy.png").is_file())
                self.assertTrue((session_path / "all_models_f1score.png").is_file())
                self.assertTrue((session_path / "accuracy_vs_f1.png").is_file())
                self.assertTrue((session_path / "best_model_confusion_matrix.png").is_file())
                self.assertTrue((session_path / "top10_models_ranking.png").is_file())
                self.assertTrue((session_path / "training_summary.json").is_file())

                with open(session_path / "training_summary.json", "r") as f:
                    summary = json.load(f)
                self.assertEqual("mock_best", summary["best_model"]["name"])
                self.assertEqual(2, len(summary["all_models"]))

    def test_schedule_periodic_training_configures_apscheduler(self):
        mock_sched_instance = MagicMock()
        mock_trigger = MagicMock()

        with patch("scheduler.BackgroundScheduler", return_value=mock_sched_instance) as mock_sched_cls, \
             patch("scheduler.CronTrigger.from_crontab", return_value=mock_trigger) as mock_trigger_from_crontab, \
             patch("scheduler.time.sleep", side_effect=KeyboardInterrupt):
            cfg = DefaultConfig(data_path=Path("dummy.csv"), registry_dir=Path("dummy_reg"))
            job_fn = MagicMock()

            schedule_periodic_training("0 */6 * * *", cfg, job_fn)

            mock_sched_cls.assert_called_once()
            mock_trigger_from_crontab.assert_called_once_with("0 */6 * * *")
            mock_sched_instance.add_job.assert_called_once_with(
                job_fn, trigger=mock_trigger, args=[cfg], id="periodic_training", replace_existing=True
            )
            mock_sched_instance.start.assert_called_once()
            mock_sched_instance.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
