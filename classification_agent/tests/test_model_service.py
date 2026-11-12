import asyncio
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier

from app.services.model_service import ModelService, PredictionError


class ModelServiceUnitTests(unittest.TestCase):
    def test_validate_input_fills_missing_values_and_orders_model_columns(self):
        service = ModelService()
        service.required_columns = ["score", "device"]
        service.numeric_cols = ["score"]
        service.categorical_cols = ["device"]

        frame = service.validate_input({"ignored": "drop", "score": "not-a-number"})

        self.assertEqual(list(frame.columns), ["score", "device"])
        self.assertEqual(frame.loc[0, "score"], 0.0)
        self.assertEqual(frame.loc[0, "device"], "unknown")

    def test_predict_uses_loaded_classifier_and_emits_label_confidence_and_hash(self):
        service = ModelService()
        service.model = DummyClassifier(strategy="most_frequent")
        service.model.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        service.classes = ["low", "high"]
        service.required_columns = ["score"]
        service.numeric_cols = ["score"]

        labels, confidences, input_hashes = service.predict([{"score": 0.7}, {"score": 0.2}, {"score": 0.7}])

        self.assertEqual(labels, ["low", "low", "low"])
        self.assertEqual(confidences, [1.0, 1.0, 1.0])
        self.assertEqual(3, len(input_hashes))
        for value in input_hashes:
            self.assertRegex(value, r"^[0-9a-f]{16}$")
        self.assertEqual(input_hashes[0], input_hashes[2])
        self.assertNotEqual(input_hashes[0], input_hashes[1])

    def test_predict_without_loaded_model_raises_prediction_error(self):
        with self.assertRaisesRegex(PredictionError, "No model loaded"):
            ModelService().predict({"score": 0.7})

    def _write_registry(self, root: Path) -> None:
        (root / "v-test_demo").mkdir()
        with open(root / "v-test_demo" / "model.pkl", "wb") as handle:
            pickle.dump({"model": {"kind": "stub"}, "preprocessor": None}, handle)
        (root / "latest.json").write_text(
            '{"tag":"v-test","classes":["safe"],"required_columns":["score","device_type"],"saved_model_name":"demo"}'
        )

    def test_load_latest_model_reads_registry_and_loads_artifact(self):
        service = ModelService()
        with tempfile.TemporaryDirectory() as directory:
            self._write_registry(Path(directory))
            with patch.dict(os.environ, {"QOPSEC_MODEL_REGISTRY": directory}):
                loaded = asyncio.run(service.load_latest_model())

        self.assertTrue(loaded)
        self.assertEqual(service.model, {"kind": "stub"})
        self.assertEqual(service.classes, ["safe"])
        self.assertEqual(service.required_columns, ["score", "device_type"])
        self.assertEqual(service.categorical_cols, ["device_type"])
        self.assertEqual(service.numeric_cols, ["score"])
        self.assertEqual(service.model_name, "demo")
        self.assertEqual(service.model_version, "v-test")

    def test_reload_is_skipped_when_registry_is_unchanged_unless_forced(self):
        service = ModelService()
        with tempfile.TemporaryDirectory() as directory:
            self._write_registry(Path(directory))
            with patch.dict(os.environ, {"QOPSEC_MODEL_REGISTRY": directory}):
                self.assertTrue(asyncio.run(service.load_latest_model()))
                self.assertFalse(asyncio.run(service.load_latest_model()))
                self.assertTrue(asyncio.run(service.load_latest_model(force=True)))
