import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from service import ImprovedRLEngineService


class RLEngineAPIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.service = ImprovedRLEngineService(
            registry_path=Path(self.temp_directory.name) / "registry.json"
        )
        self.original_service = main.rl_service
        main.rl_service = self.service
        self.addCleanup(setattr, main, "rl_service", self.original_service)
        self.client = TestClient(main.app)

    def test_health_endpoint_reports_service_version_and_policy(self):
        response = self.client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual("healthy", response.json()["status"])
        self.assertEqual("2.0.0", response.json()["version"])

    def test_act_validates_request_builds_payload_and_calls_handshake(self):
        handshake_result = {"success": True, "latency": 8.0, "resource_usage": 0.25}
        with patch.object(main, "send_to_handshake", return_value=handshake_result) as handshake:
            response = self.client.post("/act", json={
                "request_id": "integration-1",
                "source": "node-a",
                "destination": "node-b",
                "security_level": "HIGH",
                "risk_score": 0.8,
                "conf_score": 0.7,
                "dst_props": {"hardware": ["PQC"]},
            })

        self.assertEqual(200, response.status_code)
        result = response.json()
        self.assertEqual("integration-1", result["request_id"])
        self.assertEqual("integration-1", result["payload"]["requestId"])
        self.assertIn("Kyber768", result["payload"]["proposed"])
        self.assertEqual(handshake_result, result["negotiation"])
        self.assertEqual("integration-1", handshake.call_args.args[1]["request_id"])
        self.assertEqual(1, self.service.get_metrics()["successful_negotiations"])

    def test_invalid_risk_score_is_rejected_before_handshake(self):
        with patch.object(main, "send_to_handshake") as handshake:
            response = self.client.post("/act", json={
                "source": "node-a",
                "destination": "node-b",
                "security_level": "HIGH",
                "risk_score": 1.5,
            })

        self.assertEqual(422, response.status_code)
        handshake.assert_not_called()

    def test_feedback_endpoint_updates_service_and_returns_acknowledgement(self):
        self.service.decide_algorithms({"security_level": "MODERATE"})

        response = self.client.post("/feedback", json={
            "request_id": "integration-feedback",
            "success": True,
            "latency": 15.0,
            "resource_usage": 0.3,
        })

        self.assertEqual(200, response.status_code)
        self.assertEqual("feedback_processed", response.json()["status"])
        self.assertEqual(1, self.service.get_metrics()["successful_negotiations"])

    def test_handshake_failure_is_returned_as_failed_negotiation(self):
        with patch.object(main, "send_to_handshake", return_value={
            "status": "failed", "error": "handshake unavailable"
        }):
            response = self.client.post("/act", json={
                "request_id": "integration-failure",
                "source": "node-a",
                "destination": "node-b",
                "security_level": "MODERATE",
            })

        self.assertEqual(200, response.status_code)
        self.assertEqual("failed", response.json()["negotiation"]["status"])
        self.assertEqual(0, self.service.get_metrics()["successful_negotiations"])
        self.assertEqual(0, self.service.get_metrics()["failed_negotiations"])


if __name__ == "__main__":
    unittest.main()
