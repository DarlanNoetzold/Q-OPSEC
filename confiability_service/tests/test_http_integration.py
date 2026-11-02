import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from controllers import confidentiality_controller as controller
from controllers.dataset_controller import dataset_bp
from models.schemas import ContentConfidentiality, TrainResponse


class ConfidentialityHttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.models_dir = self.root / "models"
        self.metrics_dir = self.models_dir / "metrics"
        self.data_dir.mkdir()
        self.models_dir.mkdir()
        self.metrics_dir.mkdir()

        app = Flask(__name__)
        app.register_blueprint(controller.conf_bp)
        app.register_blueprint(dataset_bp)
        self.client = app.test_client()

        self.data_patch = patch("controllers.dataset_controller.DATA_DIR", self.data_dir)
        self.models_patch = patch("controllers.confidentiality_controller.MODELS_DIR", str(self.models_dir))
        self.metrics_patch = patch("controllers.confidentiality_controller.METRICS_DIR", self.metrics_dir)
        self.data_patch.start()
        self.models_patch.start()
        self.metrics_patch.start()
        self.service_patch = patch.object(controller, "_service")
        self.service = self.service_patch.start()

    def tearDown(self):
        self.service_patch.stop()
        self.metrics_patch.stop()
        self.models_patch.stop()
        self.data_patch.stop()
        self.temp_dir.cleanup()

    def test_health_and_classification_request_validation(self):
        response = self.client.get("/confidentiality/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "service": "confidentiality"})

        response = self.client.post("/confidentiality/classify", json={"request_id": "missing-pointer"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "VALIDATION_ERROR")

    def test_classify_success_and_model_not_ready_are_serialized(self):
        payload = {"request_id": "integration-1", "content_pointer": {"sample_text": "public release"}}
        expected = ContentConfidentiality(
            classification="public", score=0.7, tags=[], detected_patterns=[],
            dlp_findings=[], model_version="integration-model"
        )

        self.service.classify.return_value = expected
        response = self.client.post("/confidentiality/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["classification"], "public")
        self.assertEqual(response.get_json()["requestId"], "integration-1")

        self.service.classify.return_value = None
        response = self.client.post("/confidentiality/classify", json=payload)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "MODEL_NOT_READY")

    def test_train_validates_and_returns_service_result(self):
        response = self.client.post("/confidentiality/train", json={"n_per_class": 2})
        self.assertEqual(response.status_code, 400)
        self.service.train.assert_not_called()

        expected = TrainResponse(model_version="baseline", metrics={"accuracy": 0.9}, samples=12)
        self.service.train.return_value = expected
        response = self.client.post("/confidentiality/train", json={"n_per_class": 20, "seed": 5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model_version"], "baseline")
        self.service.train.assert_called_once()

    def test_dataset_create_upload_preview_metadata_and_delete_lifecycle(self):
        response = self.client.post("/datasets", json={"name": "integration-data"})
        self.assertEqual(response.status_code, 201)

        response = self.client.post(
            "/datasets/integration-data/files",
            data={"file": (BytesIO(b"name,score\nAda,10\n"), "sample.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)

        response = self.client.get("/datasets/integration-data/preview?file=sample.csv&n=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["rows"], [{"name": "Ada", "score": 10}])

        response = self.client.post("/datasets/integration-data/metadata", json={"owner": "qa"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.client.get("/datasets/integration-data/metadata").get_json(),
            {"metadata": {"owner": "qa"}},
        )

        response = self.client.delete("/datasets/integration-data")
        self.assertEqual(response.status_code, 200)
        self.assertFalse((self.data_dir / "integration-data").exists())

    def test_metrics_endpoint_reads_real_summary_from_isolated_directory(self):
        session = self.metrics_dir / "training_fixture"
        session.mkdir()
        (session / "training_summary.json").write_text(
            json.dumps({"timestamp": "2026-01-01", "best_model": {"name": "fixture"}}),
            encoding="utf-8",
        )

        response = self.client.get("/confidentiality/metrics/training_fixture")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["summary"]["best_model"]["name"], "fixture")
        self.assertEqual(self.client.get("/confidentiality/metrics/not-found").status_code, 404)


if __name__ == "__main__":
    unittest.main()
