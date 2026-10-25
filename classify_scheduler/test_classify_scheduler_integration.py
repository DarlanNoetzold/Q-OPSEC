import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from configs import DefaultConfig
from main import run_train_once


class ClassifySchedulerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small valid synthetic dataset for integration testing
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base_path = Path(cls.temp_dir.name)
        cls.data_path = cls.base_path / "test_dataset.csv"
        cls.registry_path = cls.base_path / "test_registry"
        cls.metrics_path = cls.base_path / "models" / "metrics"

        n_samples = 60
        labels = ["Critical", "High", "Medium", "Low"] * 15
        df = pd.DataFrame({
            "id": range(n_samples),
            "request_id_resolved": [f"req_{i}" for i in range(n_samples)],
            "created_at": ["2026-01-01T00:00:00"] * n_samples,
            "risk_score": [0.1 + (i % 8) * 0.1 for i in range(n_samples)],
            "conf_score": [0.2 + (i % 5) * 0.15 for i in range(n_samples)],
            "src_geo": ["US", "BR", "EU"] * 20,
            "security_level_label": labels,
        })
        df.to_csv(cls.data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_run_train_once_end_to_end_pipeline(self):
        cfg = DefaultConfig(
            data_path=self.data_path,
            registry_dir=self.registry_path,
            seed=42,
            test_size=0.25,
            cv_splits=2,
            max_models=2,
            target_col="security_level_label",
        )

        with patch("scheduler.METRICS_ROOT", self.metrics_path):
            run_train_once(cfg)

        # 1. Verify Model Registry outputs
        self.assertTrue(self.registry_path.exists())
        latest_file = self.registry_path / "latest.json"
        self.assertTrue(latest_file.is_file(), "latest.json was not created")

        with open(latest_file, "r") as f:
            latest = json.load(f)

        self.assertIn("saved_model_name", latest)
        self.assertIn("artifact_path", latest)
        self.assertIn("dir", latest)
        self.assertTrue(os.path.exists(latest["artifact_path"]))

        model_dir = Path(latest["dir"])
        self.assertTrue((model_dir / "model.pkl").is_file())
        self.assertTrue((model_dir / "metrics.json").is_file())
        self.assertTrue((model_dir / "candidates.json").is_file())

        with open(model_dir / "metrics.json", "r") as f:
            metrics_data = json.load(f)
        self.assertIn("cv_metrics", metrics_data)
        self.assertIn("holdout_metrics", metrics_data)
        self.assertIn("meta", metrics_data)
        self.assertEqual(metrics_data["meta"]["n_samples"], 60)

        # 2. Verify Generated Plots and Metrics
        session_id = metrics_data["meta"]["session_id"]
        session_dir = self.metrics_path / session_id
        self.assertTrue(session_dir.is_dir())
        self.assertTrue((session_dir / "all_models_accuracy.png").is_file())
        self.assertTrue((session_dir / "all_models_f1score.png").is_file())
        self.assertTrue((session_dir / "accuracy_vs_f1.png").is_file())
        self.assertTrue((session_dir / "best_model_confusion_matrix.png").is_file())
        self.assertTrue((session_dir / "top10_models_ranking.png").is_file())
        self.assertTrue((session_dir / "training_summary.json").is_file())

    def test_cli_execution_train_once_subshell(self):
        # Execute main.py via the environment Python interpreter as CLI
        cli_registry = self.base_path / "cli_registry"
        python_bin = sys.executable

        cmd = [
            python_bin,
            "main.py",
            "--mode", "train-once",
            "--data", str(self.data_path),
            "--registry", str(cli_registry),
            "--seed", "42",
            "--test-size", "0.25",
            "--cv-splits", "2",
            "--max-models", "1",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(0, result.returncode, f"CLI exited with error:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        self.assertIn("[INFO] Best model:", result.stdout)
        self.assertIn("[REGISTRY] Saved best model", result.stdout)
        self.assertTrue((cli_registry / "latest.json").is_file())


if __name__ == "__main__":
    unittest.main()
