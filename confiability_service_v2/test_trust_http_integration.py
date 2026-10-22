import unittest
from unittest.mock import patch

from app import create_app
from bootstrap_v2 import reset_trust_engine


class TrustHttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        reset_trust_engine()
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        reset_trust_engine()

    def test_evaluate_then_read_entity_and_source_history(self):
        response = self.client.post(
            "/api/v2/trust/evaluate",
            json={
                "payload": {"event": "login", "result": "success"},
                "metadata": {
                    "entity_id": "integration-user",
                    "source_id": "integration-auth",
                    "timestamp": "2025-01-01T12:00:00Z",
                    "data_type": "event",
                },
            },
        )

        self.assertEqual(200, response.status_code, response.get_json())
        result = response.get_json()
        self.assertGreaterEqual(result["trust_score"], 0.0)
        self.assertLessEqual(result["trust_score"], 1.0)

        entity_response = self.client.get("/api/v2/trust/history/integration-user")
        source_response = self.client.get("/api/v2/trust/history/source/integration-auth")

        self.assertEqual(200, entity_response.status_code)
        self.assertEqual(1, entity_response.get_json()["total"])
        self.assertEqual("integration-user", entity_response.get_json()["entity_id"])
        self.assertEqual(200, source_response.status_code)
        self.assertEqual(1, source_response.get_json()["total"])
        self.assertEqual("integration-auth", source_response.get_json()["source_id"])

    def test_evaluate_rejects_empty_payload_without_calling_orchestrator(self):
        with patch("api.v2.trust_controller.get_trust_orchestrator") as get_orchestrator:
            response = self.client.post(
                "/api/v2/trust/evaluate",
                json={"payload": {}, "metadata": {"source_id": "invalid"}},
            )

        self.assertEqual(400, response.status_code)
        get_orchestrator.assert_not_called()

    def test_health_endpoint_reports_loaded_signals(self):
        response = self.client.get("/api/v2/trust/health")

        self.assertEqual(200, response.status_code)
        body = response.get_json()
        self.assertEqual("healthy", body["status"])
        self.assertGreater(body["signals"]["total"], 0)
        self.assertEqual(body["signals"]["total"], len(body["signals"]["enabled"]))


if __name__ == "__main__":
    unittest.main()
