import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models.model_manager import ModelManager
from src.api.routes.prediction import get_manager


class FixedProbabilityModel:
    def predict_proba(self, frame):
        probabilities = np.clip(frame["amount"].to_numpy(dtype=float) / 100, 0, 1)
        return np.column_stack((1 - probabilities, probabilities))


class PredictionApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.manager.feature_names = ["amount", "channel"]
        self.manager.predict.return_value = {
            "version": "v-test",
            "n_records": 1,
            "models": {"forest": {"probabilities": [0.1, 0.9], "predictions": [0, 1]}},
        }
        self._dependency_overrides_before_test = dict(app.dependency_overrides)
        app.dependency_overrides[get_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self._dependency_overrides_before_test)

    def test_single_prediction_runs_through_http_validation_and_manager(self):
        response = self.client.post(
            "/predict/",
            json={"single": {"features": {"amount": 17, "channel": "mobile"}}, "models": ["forest"], "request_id": "req-1"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["request_id"], "req-1")
        self.assertEqual(response.json()["n_records"], 1)
        self.manager.predict.assert_called_once_with([{"amount": 17, "channel": "mobile"}], model_names=["forest"])

    def test_batch_prediction_unwraps_feature_envelopes(self):
        response = self.client.post(
            "/predict/",
            json={"batch": {"records": [{"features": {"amount": 3}}, {"amount": 9}]}, "models": ["forest"]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.manager.predict.assert_called_once_with([{"amount": 3}, {"amount": 9}], model_names=["forest"])

    def test_request_without_single_or_batch_records_returns_bad_request(self):
        response = self.client.post("/predict/", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("single", response.json()["detail"])
        self.manager.predict.assert_not_called()

    def test_invalid_json_shape_is_rejected_by_request_validation(self):
        response = self.client.post("/predict/", json={"single": {"amount": 2}})

        self.assertEqual(response.status_code, 422)
        self.manager.predict.assert_not_called()

    def test_empty_batch_returns_empty_prediction_result(self):
        self.manager.predict.return_value = {"version": "v-test", "n_records": 0, "models": {}}

        response = self.client.post("/predict/", json={"batch": {"records": []}})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["n_records"], 0)
        self.manager.predict.assert_called_once_with([], model_names=None)

    def test_requested_version_is_loaded_before_prediction(self):
        response = self.client.post("/predict/", json={"single": {"features": {"amount": 17}}, "version": "v2"})

        self.assertEqual(response.status_code, 200, response.text)
        self.manager.load.assert_called_once_with("v2")
        self.manager.predict.assert_called_once()

    def test_model_version_load_failure_returns_internal_error(self):
        self.manager.load.side_effect = FileNotFoundError("version absent")

        with patch("src.api.routes.prediction.logger.exception"):
            response = self.client.post("/predict/", json={"single": {"features": {"amount": 17}}, "version": "missing"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to load model version missing", response.json()["detail"])
        self.manager.predict.assert_not_called()

    def test_health_endpoint_is_available(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_http_prediction_uses_real_model_manager_and_serializes_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir) / "models"
            version_dir = models_root / "v1"
            version_dir.mkdir(parents=True)
            (version_dir / "feature_names.json").write_text(json.dumps(["amount"]))
            (version_dir / "demo_metadata.json").write_text(json.dumps({"optimal_threshold": 0.5}))
            joblib.dump(FixedProbabilityModel(), version_dir / "demo_model.pkl")
            manager = ModelManager(models_root=str(models_root), eval_root=str(Path(tmp_dir) / "evaluation"))

            app.dependency_overrides[get_manager] = lambda: manager
            response = self.client.post(
                "/predict/",
                json={"batch": {"records": [{"amount": 20}, {"amount": 80}]}, "models": ["demo"]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["version"], "v1")
        self.assertEqual(payload["n_records"], 2)
        self.assertEqual(payload["models"]["demo"]["probabilities"], [0.2, 0.8])
        self.assertEqual(payload["models"]["demo"]["predictions"], [0, 1])


if __name__ == "__main__":
    unittest.main()
