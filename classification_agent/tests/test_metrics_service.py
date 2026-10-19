import asyncio
import unittest
from unittest.mock import patch

from app.services.metrics_service import MetricsService


class MetricsServiceTests(unittest.TestCase):
    def test_metrics_aggregate_requests_errors_and_prediction_batches(self):
        service = MetricsService()
        service.record_request("/predict", "POST", 200, 0.025)
        service.record_request("/predict", "POST", 500, 0.075)
        service.record_prediction("classifier-v1", batch_size=3)
        service.record_model_reload("classifier-v1")

        metrics = service.get_metrics()

        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["total_predictions"], 3)
        self.assertEqual(metrics["error_rate"], 0.5)
        self.assertEqual(metrics["average_response_time_ms"], 50.0)
        self.assertEqual(metrics["model_reload_count"], 1)
        self.assertEqual(metrics["current_model"], "classifier-v1")

    def test_reset_metrics_clears_counters_and_model(self):
        service = MetricsService()
        service.record_request("/health", "GET", 200, 0.01)
        service.record_prediction("classifier-v1")

        service.reset_metrics()

        self.assertEqual(service.get_metrics()["total_requests"], 0)
        self.assertEqual(service.get_metrics()["total_predictions"], 0)
        self.assertIsNone(service.get_metrics()["current_model"])


class MetricsAndPredictionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_predict_service_returns_predictions_and_records_batch_metrics(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from app.api.v1.endpoints import PredictionRequest, predict
        from app.core.config import settings
        from app.utils.exceptions import setup_exception_handlers

        class ModelStub:
            model_name = "classifier-v1"
            model_version = "1.2"
            classes = ["safe", "risky"]

            @staticmethod
            def is_model_loaded():
                return True

            @staticmethod
            def validate_input(data):
                return None

            @staticmethod
            def predict(data):
                count = len(data) if isinstance(data, list) else 1
                return ["safe"] * count, [0.9] * count, ["hash"] * count

        metrics = MetricsService()
        app = FastAPI()
        setup_exception_handlers(app)

        async def endpoint(request: Request):
            return await predict(
                PredictionRequest(data=[{"score": 0.1}, {"score": 0.2}]), request, {"user": "test"}
            )

        app.add_api_route("/predict", endpoint, methods=["POST"])
        with patch("app.api.v1.endpoints.model_service", ModelStub()), \
             patch("app.api.v1.endpoints.metrics_service", metrics):
            response = TestClient(app).post("/predict")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["batch_size"], 2)
        self.assertEqual([item["label"] for item in body["results"]], ["safe", "safe"])
        self.assertEqual(metrics.get_metrics()["total_predictions"], 2)
