import unittest

from models import DecryptRequest, EncryptRequest


class CryptoRequestValidationTests(unittest.TestCase):
    def test_encrypt_rejects_blank_algorithm(self):
        with self.assertRaises(ValueError):
            EncryptRequest(algorithm="   ")

    def test_encrypt_rejects_blank_optional_identifiers(self):
        with self.assertRaises(ValueError):
            EncryptRequest(
                algorithm="AES256_GCM",
                session_id="session-1",
                request_id="   ",
            )

    def test_decrypt_rejects_blank_core_text_fields(self):
        valid = {
            "session_id": "session-1",
            "request_id": "request-1",
            "algorithm": "AES256_GCM",
            "nonce_b64": "bm9uY2U=",
            "ciphertext_b64": "Y2lwaGVydGV4dA==",
        }
        for field in ("session_id", "request_id", "algorithm"):
            payload = valid.copy()
            payload[field] = "   "
            with self.subTest(field=field), self.assertRaises(ValueError):
                DecryptRequest(**payload)


if __name__ == "__main__":
    unittest.main()
