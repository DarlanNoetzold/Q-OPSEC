import base64
import unittest

from crypto.classical import derive_session_key, generate_classical_key


class ClassicalCryptoUnitTests(unittest.TestCase):
    def test_aes256_key_is_valid_base64_with_32_decoded_bytes(self):
        encoded_key = generate_classical_key("AES256_GCM")

        self.assertEqual(32, len(base64.b64decode(encoded_key, validate=True)))

    def test_aes128_key_is_valid_base64_with_16_decoded_bytes(self):
        encoded_key = generate_classical_key("AES128_GCM")

        self.assertEqual(16, len(base64.b64decode(encoded_key, validate=True)))

    def test_session_key_derivation_is_repeatable_for_same_context(self):
        raw_material = b"unit-test-secret-material"

        first = derive_session_key(raw_material, length=32, context=b"session-a")
        second = derive_session_key(raw_material, length=32, context=b"session-a")

        self.assertEqual(first, second)
        self.assertEqual(32, len(first))

    def test_session_key_derivation_separates_contexts(self):
        raw_material = b"unit-test-secret-material"

        key_a = derive_session_key(raw_material, length=32, context=b"session-a")
        key_b = derive_session_key(raw_material, length=32, context=b"session-b")

        self.assertNotEqual(key_a, key_b)

    def test_unknown_symmetric_algorithm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported classical algorithm"):
            generate_classical_key("NOT_A_REAL_ALGORITHM")


if __name__ == "__main__":
    unittest.main()
