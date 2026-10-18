import base64
import unittest
from unittest.mock import AsyncMock, patch

from cryptography.exceptions import InvalidTag

from crypto_engine import aead_decrypt, aead_encrypt, b64d, b64e


class CryptoEngineUnitTests(unittest.TestCase):
    def setUp(self):
        self.ctx = {"key_material": base64.b64encode(b"unit-test-session-key-material").decode()}

    def test_base64_helpers_round_trip_binary_data(self):
        payload = bytes(range(256))
        self.assertEqual(payload, b64d(b64e(payload)))

    def test_aead_algorithms_round_trip_plaintext_and_aad(self):
        plaintext = b"sensitive payload"
        aad = b"session:unit|request:1"

        for algorithm in ("AES256_GCM", "CHACHA20_POLY1305"):
            with self.subTest(algorithm=algorithm):
                nonce_b64, ciphertext_b64 = aead_encrypt(self.ctx, algorithm, plaintext, aad)
                self.assertEqual(
                    plaintext,
                    aead_decrypt(self.ctx, algorithm, nonce_b64, ciphertext_b64, aad),
                )

    def test_aead_rejects_modified_associated_data(self):
        nonce_b64, ciphertext_b64 = aead_encrypt(
            self.ctx, "AES256_GCM", b"sensitive payload", b"original metadata"
        )

        with self.assertRaises(InvalidTag):
            aead_decrypt(
                self.ctx, "AES256_GCM", nonce_b64, ciphertext_b64, b"modified metadata"
            )

    def test_missing_session_id_fails_before_network_access(self):
        from crypto_engine import fetch_key_context
        import asyncio

        with patch("crypto_engine.httpx.AsyncClient") as client_factory:
            with self.assertRaisesRegex(ValueError, "session_id is required"):
                asyncio.run(fetch_key_context(None, None))
        client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
