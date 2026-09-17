import unittest
from unittest.mock import patch

from key_manager import build_session


class StrictModeRegressionTests(unittest.TestCase):
    def test_strict_mode_rejects_when_requested_algorithm_is_fallback(self):
        with patch(
            "key_manager.generate_key",
            return_value=("AES256_GCM", "material", "classical", None),
        ):
            with self.assertRaisesRegex(ValueError, "strict mode"):
                build_session(
                    session_id="session-strict",
                    request_id="request-strict",
                    algorithm="Kyber768",
                    ttl_seconds=60,
                    strict=True,
                )

    def test_non_strict_mode_keeps_fallback_behavior(self):
        with patch(
            "key_manager.generate_key",
            return_value=("AES256_GCM", "material", "classical", None),
        ):
            result = build_session(
                session_id="session-fallback",
                request_id="request-fallback",
                algorithm="Kyber768",
                ttl_seconds=60,
                strict=False,
            )

        self.assertEqual("AES256_GCM", result[2])
        self.assertTrue(result[5])


if __name__ == "__main__":
    unittest.main()
