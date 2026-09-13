import base64
import unittest

from crypto_engine import _derive_aead_key


class CryptoAlgorithmCompatibilityTests(unittest.TestCase):
    def test_pqc_selection_uses_aes_for_payload_encryption(self):
        ctx = {"key_material": base64.b64encode(b"pqc-session-material").decode()}
        key, nonce_size = _derive_aead_key(ctx, "Dilithium3", for_encrypt=True)
        self.assertEqual(32, len(key))
        self.assertEqual(12, nonce_size)

    def test_classical_aead_algorithms_keep_existing_behavior(self):
        ctx = {"key_material": base64.b64encode(b"session-material").decode()}
        for algorithm in ("AES256_GCM", "CHACHA20_POLY1305"):
            key, nonce_size = _derive_aead_key(ctx, algorithm, for_encrypt=True)
            self.assertEqual(32, len(key))
            self.assertEqual(12, nonce_size)


if __name__ == "__main__":
    unittest.main()
