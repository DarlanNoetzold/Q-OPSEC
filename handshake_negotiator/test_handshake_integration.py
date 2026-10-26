import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import httpx

import main


class HandshakeNegotiatorIntegrationTests(unittest.TestCase):
    """
    Testes de integração para a API do Handshake Negotiator, cobrindo endpoints HTTP
    e orquestração das chamadas externas para KMS, KDE, Crypto e Validation.
    """

    def setUp(self):
        self.client = TestClient(main.app)
        self.mock_kms_response = {
            "session_id": "sess-test-12345",
            "request_id": "req-test-99999",
            "selected_algorithm": "Kyber1024",
            "key_material": "bW9ja19rZXlfbWF0ZXJpYWxfMTIzNDU=",
            "expires_at": datetime.now(timezone.utc).isoformat(),
            "fallback_applied": False,
            "fallback_reason": None,
            "source_of_key": "pqc"
        }
        self.mock_kde_response = {
            "status": "DELIVERED",
            "delivery_id": "del-888"
        }
        self.mock_crypto_response = {
            "nonce_b64": "bW9ja19ub25jZQ==",
            "ciphertext_b64": "bW9ja19jaXBoZXJ0ZXh0",
            "algorithm": "Kyber1024",
            "expires_at": 1708012800
        }
        self.mock_validation_response = {
            "status": "FORWARDED",
            "validation_id": "val-777"
        }

    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual("healthy", data["status"])
        self.assertEqual("handshake-negotiator", data["service"])
        self.assertIn("kms", data["integrations"])
        self.assertIn("kde", data["integrations"])
        self.assertIn("crypto", data["integrations"])
        self.assertIn("validation", data["integrations"])

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual("Handshake Negotiator", data["service"])
        self.assertEqual("2.3.0", data["version"])
        self.assertIn("/handshake", data["endpoints"].values())

    @patch("main.httpx.AsyncClient")
    def test_handshake_orchestration_happy_path(self, mock_client_cls):
        # Configure mock async client to respond to KMS, KDE, Crypto, Validation
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        def post_side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "kms" in url:
                resp.json.return_value = self.mock_kms_response
                resp.text = str(self.mock_kms_response)
            elif "deliver" in url:
                resp.json.return_value = self.mock_kde_response
                resp.text = str(self.mock_kde_response)
            elif "encrypt" in url:
                resp.json.return_value = self.mock_crypto_response
                resp.text = str(self.mock_crypto_response)
            elif "validation" in url:
                resp.json.return_value = self.mock_validation_response
                resp.text = str(self.mock_validation_response)
            return resp

        mock_client.post.side_effect = post_side_effect

        payload = {
            "request_id": "req-test-99999",
            "source": "sender-node",
            "destination": "http://192.168.18.18:9000",
            "proposed": ["Kyber1024", "AES256_GCM"],
            "dst_props": {"algorithms": ["Kyber1024", "AES256_GCM"]}
        }

        response = self.client.post("/handshake", json=payload)
        self.assertEqual(200, response.status_code, response.text)
        data = response.json()

        self.assertEqual("req-test-99999", data["request_id"])
        self.assertEqual("sess-test-12345", data["session_id"])
        self.assertEqual("Kyber1024", data["selected_algorithm"])
        self.assertEqual(self.mock_kms_response["key_material"], data["key_material"])
        self.assertEqual("bW9ja19ub25jZQ==", data["crypto_nonce_b64"])
        self.assertEqual("bW9ja19jaXBoZXJ0ZXh0", data["crypto_ciphertext_b64"])
        self.assertIn("Negotiation completed successfully", data["message"])
        self.assertIn("kde", data["delivery_status"])
        self.assertIn("validation", data["delivery_status"])

    @patch("main.httpx.AsyncClient")
    def test_handshake_kms_failure_returns_500(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        failing_resp = MagicMock()
        failing_resp.status_code = 503
        failing_resp.text = "KMS Unavailable"
        mock_client.post.return_value = failing_resp

        payload = {
            "source": "sender-node",
            "destination": "http://192.168.18.18:9000",
            "proposed": ["Kyber1024"]
        }

        response = self.client.post("/handshake", json=payload)
        self.assertEqual(500, response.status_code)
        self.assertIn("Erro ao criar chave no KMS", response.json()["detail"])

    @patch("main.httpx.AsyncClient")
    def test_handshake_resilience_when_crypto_returns_non_200(self, mock_client_cls):
        # Even if crypto fails or returns non-200, handshake should complete and report status
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        def post_side_effect(url, **kwargs):
            resp = MagicMock()
            if "kms" in url:
                resp.status_code = 200
                resp.json.return_value = self.mock_kms_response
                resp.text = str(self.mock_kms_response)
            elif "deliver" in url:
                resp.status_code = 200
                resp.json.return_value = self.mock_kde_response
                resp.text = str(self.mock_kde_response)
            elif "encrypt" in url:
                resp.status_code = 404
                resp.text = "Message not found in interceptor"
            return resp

        mock_client.post.side_effect = post_side_effect

        payload = {
            "source": "sender-node",
            "destination": "http://192.168.18.18:9000",
            "proposed": ["Kyber1024"]
        }

        response = self.client.post("/handshake", json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIsNone(data["crypto_nonce_b64"])
        self.assertIsNone(data["crypto_ciphertext_b64"])
        self.assertIn("no crypto output available", data["delivery_status"])

    def test_handshake_validation_rejects_missing_required_fields(self):
        # Missing source and destination
        response = self.client.post("/handshake", json={"proposed": ["AES256_GCM"]})
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
