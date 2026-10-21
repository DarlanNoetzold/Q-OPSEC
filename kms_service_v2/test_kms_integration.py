import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from storage import KeySessionStore


class KMSHttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = KeySessionStore()
        self.store_patch = patch("main.session_store", self.store)
        self.store_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.store_patch.stop()

    def test_create_retrieve_and_delete_key_session(self):
        created = self.client.post(
            "/kms/create_key",
            json={
                "session_id": "integration-session",
                "request_id": "integration-request",
                "algorithm": "AES256_GCM",
                "ttl_seconds": 120,
            },
        )

        self.assertEqual(200, created.status_code, created.text)
        payload = created.json()
        self.assertEqual("integration-session", payload["session_id"])
        self.assertEqual("integration-request", payload["request_id"])
        self.assertEqual("AES256_GCM", payload["selected_algorithm"])
        self.assertEqual("classical", payload["source_of_key"])
        self.assertTrue(payload["key_material"])

        by_session = self.client.get("/kms/get_key/integration-session")
        by_request = self.client.get(
            "/kms/get_key", params={"request_id": "integration-request"}
        )
        self.assertEqual(200, by_session.status_code)
        self.assertEqual(200, by_request.status_code)
        self.assertEqual(payload["key_material"], by_session.json()["key_material"])
        self.assertEqual(payload["key_material"], by_request.json()["key_material"])

        deleted = self.client.delete("/kms/session/integration-session")
        self.assertEqual(200, deleted.status_code)
        self.assertEqual(404, self.client.get("/kms/get_key/integration-session").status_code)

    def test_create_key_rejects_invalid_request_without_saving_session(self):
        response = self.client.post(
            "/kms/create_key",
            json={"session_id": "bad-session", "algorithm": "AES256_GCM", "ttl_seconds": 0},
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual(0, self.store.stats()["total_sessions"])

    def test_health_endpoint_reports_healthy_service(self):
        response = self.client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual("healthy", response.json()["status"])
        self.assertEqual("memory", response.json()["components"]["storage"]["backend"])


if __name__ == "__main__":
    unittest.main()
