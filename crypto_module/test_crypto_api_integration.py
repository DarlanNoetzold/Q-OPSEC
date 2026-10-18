import base64
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from crypto_engine import b64e


class CryptoApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.ctx = {
            "session_id": "integration-session",
            "request_id": "integration-request",
            "key_material": base64.b64encode(b"integration-session-key-material").decode(),
            "expires_at": 2_000_000_000,
        }

    @patch("main.fetch_key_context")
    def test_encrypt_and_decrypt_round_trip_through_http_api(self, fetch_context):
        fetch_context.return_value = self.ctx
        plaintext = b"integration payload"
        aad = b"integration aad"

        encrypted = self.client.post(
            "/encrypt",
            json={
                "session_id": self.ctx["session_id"],
                "request_id": self.ctx["request_id"],
                "algorithm": "CHACHA20_POLY1305",
                "plaintext_b64": b64e(plaintext),
                "aad_b64": b64e(aad),
                "fetch_from_interceptor": False,
            },
        )

        self.assertEqual(200, encrypted.status_code, encrypted.text)
        encrypted_body = encrypted.json()
        self.assertEqual(self.ctx["session_id"], encrypted_body["session_id"])
        self.assertEqual("CHACHA20_POLY1305", encrypted_body["algorithm"])
        self.assertEqual(self.ctx["expires_at"], encrypted_body["expires_at"])

        decrypted = self.client.post(
            "/decrypt",
            json={
                "session_id": self.ctx["session_id"],
                "request_id": self.ctx["request_id"],
                "algorithm": encrypted_body["algorithm"],
                "nonce_b64": encrypted_body["nonce_b64"],
                "ciphertext_b64": encrypted_body["ciphertext_b64"],
                "aad_b64": b64e(aad),
            },
        )

        self.assertEqual(200, decrypted.status_code, decrypted.text)
        self.assertEqual(b64e(plaintext), decrypted.json()["plaintext_b64"])
        self.assertEqual(2, fetch_context.call_count)

    @patch("main.fetch_key_context")
    def test_encrypt_reports_invalid_request_as_bad_request(self, fetch_context):
        fetch_context.return_value = self.ctx

        response = self.client.post(
            "/encrypt",
            json={
                "session_id": self.ctx["session_id"],
                "algorithm": "AES256_GCM",
                "fetch_from_interceptor": False,
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("plaintext_b64 is required", response.json()["detail"])

    def test_health_endpoint_reports_crypto_module(self):
        response = self.client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"status": "healthy", "module": "crypto", "version": "1.3.1"},
            response.json(),
        )


if __name__ == "__main__":
    unittest.main()
