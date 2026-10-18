import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, patch

import main
from crypto_engine import b64e
from models import DecryptRequest, EncryptRequest


class CryptoServiceTests(unittest.TestCase):
    def setUp(self):
        self.ctx = {
            "session_id": "service-session",
            "request_id": "service-request",
            "key_material": base64.b64encode(b"service-session-key-material").decode(),
            "expires_at": "2033-05-18T03:33:20+00:00",
        }

    @patch("main.aead_encrypt", return_value=("bm9uY2U=", "Y2lwaGVydGV4dA=="))
    @patch("main.fetch_key_context", new_callable=AsyncMock)
    def test_encrypt_uses_supplied_plaintext_and_custom_aad(self, fetch_context, encrypt):
        fetch_context.return_value = self.ctx
        request = EncryptRequest(
            session_id="requested-session",
            request_id="requested-request",
            algorithm="AES256_GCM",
            plaintext_b64=b64e(b"plain text"),
            aad_b64=b64e(b"custom aad"),
            fetch_from_interceptor=False,
        )

        result = asyncio.run(main.encrypt(request))

        self.assertEqual("service-session", result.session_id)
        self.assertEqual(2_000_000_000, result.expires_at)
        self.assertEqual(b"plain text", encrypt.call_args.args[2])
        self.assertEqual(b"custom aad", encrypt.call_args.args[3])
        fetch_context.assert_awaited_once_with(session_id="requested-session", request_id=None)

    @patch("main.fetch_message_from_interceptor", new_callable=AsyncMock)
    @patch("main.fetch_key_context", new_callable=AsyncMock)
    def test_encrypt_by_request_id_fetches_message_then_encrypts(self, fetch_context, fetch_message):
        fetch_context.return_value = self.ctx
        fetch_message.return_value = {"message": "message from interceptor"}

        result = asyncio.run(
            main.encrypt_by_request_id(
                main.EncryptByRequestId(
                    request_id="service-request",
                    session_id="requested-session",
                    algorithm="AES256_GCM",
                )
            )
        )

        self.assertEqual("service-session", result.session_id)
        fetch_message.assert_awaited_once_with("service-request")
        fetch_context.assert_awaited_once_with(session_id="requested-session", request_id=None)

    @patch("main.fetch_key_context", new_callable=AsyncMock)
    def test_decrypt_maps_invalid_authentication_to_bad_request(self, fetch_context):
        fetch_context.return_value = self.ctx
        request = DecryptRequest(
            session_id="requested-session",
            request_id="service-request",
            algorithm="AES256_GCM",
            nonce_b64="bm9uY2U=",
            ciphertext_b64="Y2lwaGVydGV4dA==",
        )

        with patch("main.aead_decrypt", side_effect=ValueError("authentication failed")):
            with self.assertRaises(main.HTTPException) as raised:
                asyncio.run(main.decrypt(request))

        self.assertEqual(400, raised.exception.status_code)
        self.assertEqual("authentication failed", raised.exception.detail)

    @patch("main.fetch_key_context", new_callable=AsyncMock)
    def test_decrypt_returns_plaintext_as_base64(self, fetch_context):
        fetch_context.return_value = self.ctx
        request = DecryptRequest(
            session_id="requested-session",
            request_id="service-request",
            algorithm="AES256_GCM",
            nonce_b64="bm9uY2U=",
            ciphertext_b64="Y2lwaGVydGV4dA==",
            aad_b64=None,
        )

        with patch("main.aead_decrypt", return_value=b"decrypted"):
            result = asyncio.run(main.decrypt(request))

        self.assertEqual(b64e(b"decrypted"), result.plaintext_b64)


if __name__ == "__main__":
    unittest.main()
