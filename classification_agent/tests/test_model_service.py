import asyncio
import unittest
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

        labels, confidences, input_hash = service.predict({"score": 0.7})

        self.assertEqual(labels, ["low"])
        self.assertEqual(confidences, [1.0])
        self.assertRegex(input_hash, r"^[0-9a-f]{8}$")

    def test_predict_without_loaded_model_raises_prediction_error(self):
        with self.assertRaisesRegex(PredictionError, "No model loaded"):
            ModelService().predict({"score": 0.7})

    def test_load_latest_model_reads_registry_and_loads_artifact(self):
        service = ModelService()
        artifact = {"model": object(), "preprocessor": None}
        latest = '{"tag":"v-test","classes":["safe"],"required_columns":["score"],"saved_model_name":"demo"}'
        model_info = {
            "tag": "v-test", "classes": ["safe"], "required_columns": ["score"],
            "saved_model_name": "demo"
        }
        path_exists = patch("app.services.model_service.Path.exists", return_value=True)
        file_open = patch("app.services.model_service.open", create=True)
        listdir = patch("app.services.model_service.os.listdir", side_effect=[["v-test"], ["model.pkl"]])
        isdir = patch("app.services.model_service.os.path.isdir", return_value=True)
        load_artifact = patch("app.services.model_service.pickle.load", return_value=artifact)
        load_json = patch("app.services.model_service.json.load", return_value=model_info)

        with path_exists, file_open as open_mock, listdir, isdir, load_artifact, load_json:
            open_mock.return_value.__enter__.return_value.read.return_value = latest
            loaded = asyncio.run(service.load_latest_model())

        self.assertTrue(loaded)
        self.assertIs(service.model, artifact["model"])
        self.assertEqual(service.classes, ["safe"])
        self.assertEqual(service.required_columns, ["score"])
        self.assertEqual(service.model_name, "demo")
        self.assertEqual(service.model_version, "v-test")
