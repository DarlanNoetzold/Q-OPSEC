import os
from pathlib import Path
import tempfile
import unittest

from src.api.models.model_manager import ModelManager


class ModelManagerArtifactTests(unittest.TestCase):
    def test_get_model_metrics_reads_version_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_root = Path(tmp_dir) / "evaluation"
            version_dir = eval_root / "v1"
            version_dir.mkdir(parents=True)
            expected = {"accuracy": 0.95, "recall": 0.91}
            (version_dir / "metrics.json").write_text('{"accuracy": 0.95, "recall": 0.91}')
            manager = ModelManager(models_root=str(Path(tmp_dir) / "models"), eval_root=str(eval_root))

            actual = manager.get_model_metrics("v1")

        self.assertEqual(actual, expected)

    def test_get_model_metrics_returns_empty_when_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelManager(models_root=str(Path(tmp_dir) / "models"), eval_root=str(Path(tmp_dir) / "evaluation"))

            self.assertEqual(manager.get_model_metrics("v1"), {})

    def test_get_dataset_summary_returns_fallback_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(tmp_dir)
                manager = ModelManager()
                actual = manager.get_dataset_summary()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(actual, "No dataset summary available")


if __name__ == "__main__":
    unittest.main()
