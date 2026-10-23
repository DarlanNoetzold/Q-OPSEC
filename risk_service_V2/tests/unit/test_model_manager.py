import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.api.models.model_manager import ModelManager


class ModelManagerUnitTests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager(models_root="unused/models", eval_root="unused/evaluation")

    def test_normalize_feature_names_orders_numeric_mapping(self):
        result = self.manager._normalize_feature_names({"1": "amount", "0": "account_age"})

        self.assertEqual(result, ["account_age", "amount"])

    def test_normalize_feature_names_uses_all_features(self):
        result = self.manager._normalize_feature_names({"all_features": ["amount", "channel"]})

        self.assertEqual(result, ["amount", "channel"])

    def test_normalize_feature_names_falls_back_to_keys(self):
        result = self.manager._normalize_feature_names({"amount": "numeric", "channel": "categorical"})

        self.assertEqual(result, ["amount", "channel"])

    def test_normalize_feature_names_coerces_scalar(self):
        self.assertEqual(self.manager._normalize_feature_names(5), ["5"])

    def test_prepare_dataframe_orders_features_and_fills_missing_values(self):
        self.manager.feature_names = ["amount", "account_age"]

        frame = self.manager._prepare_dataframe([{"account_age": 4, "extra": "ignored"}])

        self.assertEqual(frame.columns.tolist(), ["amount", "account_age"])
        self.assertTrue(np.isnan(frame.loc[0, "amount"]))
        self.assertEqual(frame.loc[0, "account_age"], 4)

    def test_prepare_dataframe_unwraps_features_envelope(self):
        self.manager.feature_names = ["amount"]

        frame = self.manager._prepare_dataframe([{"features": {"amount": 12.5}}])

        self.assertEqual(frame.loc[0, "amount"], 12.5)

    def test_load_uses_latest_version_and_reads_feature_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            version_dir = Path(tmp_dir) / "v1"
            version_dir.mkdir()
            (version_dir / "feature_names.json").write_text(json.dumps(["amount", "channel"]))
            manager = ModelManager(models_root=tmp_dir)

            result = manager.load()

        self.assertEqual(result, "v1")
        self.assertEqual(manager.loaded_version, "v1")
        self.assertEqual(manager.feature_names, ["amount", "channel"])

    def test_load_missing_models_directory_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelManager(models_root=str(Path(tmp_dir) / "absent"))

            with self.assertRaisesRegex(FileNotFoundError, "No model versions"):
                manager.load()

    def test_load_unknown_version_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelManager(models_root=tmp_dir)

            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                manager.load("missing")

    def test_list_versions_returns_sorted_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for version in ("v2", "v1"):
                (Path(tmp_dir) / version).mkdir()
            (Path(tmp_dir) / "not-a-directory").write_text("ignored")
            manager = ModelManager(models_root=tmp_dir)

            self.assertEqual(manager.list_versions(), ["v1", "v2"])

    def test_apply_label_encoders_maps_unseen_category_to_unknown(self):
        class Encoder:
            classes_ = np.array(["mobile", "web"])

            def transform(self, values):
                if "unknown" in values:
                    raise ValueError("unknown category")
                return np.array([0 if value == "mobile" else 1 for value in values])

        self.manager.label_encoders = {"channel": Encoder()}
        import pandas as pd
        frame = pd.DataFrame({"channel": ["mobile", "unknown"]})

        transformed = self.manager._apply_label_encoders(frame)

        self.assertEqual(transformed["channel"].tolist(), [0, -1])

    def test_predict_applies_threshold_and_returns_metadata(self):
        class ProbabilityModel:
            def predict_proba(self, frame):
                return np.array([[0.8, 0.2], [0.1, 0.9]])

        self.manager.loaded_version = "v-test"
        self.manager.feature_names = ["amount"]
        self.manager.models = {"demo": ProbabilityModel()}
        self.manager.metadata = {"demo": {"optimal_threshold": 0.7}}

        result = self.manager.predict([{"amount": 10}, {"amount": 20}])

        self.assertEqual(result["version"], "v-test")
        self.assertEqual(result["n_records"], 2)
        self.assertEqual(result["models"]["demo"]["probabilities"], [0.2, 0.9])
        self.assertEqual(result["models"]["demo"]["predictions"], [0, 1])
        self.assertEqual(result["models"]["demo"]["threshold"], 0.7)

    def test_predict_reports_unknown_model_without_crashing(self):
        self.manager.loaded_version = "v-test"

        result = self.manager.predict([{"amount": 10}], model_names=["missing"])

        self.assertEqual(result["models"]["missing"], {"error": "model not loaded"})


if __name__ == "__main__":
    unittest.main()
