import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import create_app
from app.api.v1.endpoints import get_current_user
from app.services.metrics_service import metrics_service


class ClassificationAgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        metrics_service.reset_metrics()
        self.app = create_app()
        self.app.dependency_overrides[get_current_user] = lambda: {"user": "integration-test"}
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.dependency_overrides.clear()
        metrics_service.reset_metrics()

    def test_health_endpoint_returns_service_health_and_request_headers(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["status"], {"healthy", "degraded", "ok"})
        self.assertIn("version", body)
        self.assertIn("model_loaded", body)
        self.assertTrue(response.headers.get("x-request-id"))
        self.assertTrue(response.headers.get("x-response-time", "").endswith("ms"))

    def test_predict_endpoint_runs_real_model_service_and_serializes_response(self):
        from app.api.v1.endpoints import model_service

        class TinyClassifier:
            def predict(self, frame):
                return [0] * len(frame)

            def predict_proba(self, frame):
                return [[0.8, 0.2] for _ in range(len(frame))]

        with patch.object(model_service, "model", TinyClassifier()), \
             patch.object(model_service, "required_columns", ["score"]), \
             patch.object(model_service, "numeric_cols", ["score"]), \
             patch.object(model_service, "categorical_cols", []), \
             patch.object(model_service, "classes", ["safe", "risky"]), \
             patch.object(model_service, "model_name", "integration-model"), \
             patch.object(model_service, "model_version", "test"):
            response = self.client.post(
                "/api/v1/predict",
                json={"data": [{"score": 0.2}, {"score": 0.9}]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["batch_size"], 2)
        self.assertEqual(body["model_name"], "integration-model")
        self.assertEqual([result["label"] for result in body["results"]], ["safe", "safe"])
        self.assertEqual([result["confidence"] for result in body["results"]], [0.8, 0.8])
        self.assertEqual(metrics_service.get_metrics()["total_predictions"], 2)

    def test_predict_without_model_returns_service_unavailable(self):
        from app.api.v1.endpoints import model_service

        with patch.object(model_service, "model", None):
            response = self.client.post("/api/v1/predict", json={"data": {"score": 0.2}})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "ModelNotLoadedException")

    def test_predict_rejects_missing_data_with_validation_response(self):
        response = self.client.post("/api/v1/predict", json={})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "validation_error")
        self.assertTrue(response.json()["details"])
