import base64
import unittest
from unittest.mock import patch

from key_manager import build_session, generate_key


class KeyManagerServiceTests(unittest.TestCase):
    def test_build_session_creates_classical_key_with_requested_identifiers(self):
        with patch("key_manager.time.time", return_value=1_700_000_000), patch(
            "key_manager.uuid.uuid4"
        ) as uuid4:
            result = build_session(
                session_id="service-session",
                request_id="service-request",
                algorithm="AES256_GCM",
                ttl_seconds=120,
            )

        self.assertEqual("service-session", result[0])
        self.assertEqual("service-request", result[1])
        self.assertEqual("AES256_GCM", result[2])
        self.assertEqual(32, len(base64.b64decode(result[3], validate=True)))
        self.assertEqual(1_700_000_120, result[4])
        self.assertFalse(result[5])
        self.assertIsNone(result[6])
        self.assertEqual("classical", result[7])
        self.assertIsNone(result[8])
        uuid4.assert_not_called()

    def test_build_session_generates_missing_identifiers(self):
        uuid_values = iter(["a" * 32, "b" * 32])

        class FakeUUID:
            def __init__(self, value):
                self.hex = value

        with patch("key_manager.uuid.uuid4", side_effect=lambda: FakeUUID(next(uuid_values))), patch(
            "key_manager.time.time", return_value=100
        ):
            result = build_session(
                session_id=None,
                request_id=None,
                algorithm="AES256_GCM",
                ttl_seconds=30,
            )

        self.assertEqual("sess_aaaaaaaaaaaaaaaa", result[0])
        self.assertEqual("req_bbbbbbbbbbbbbbbb", result[1])
        self.assertGreaterEqual(result[4], 130)

    def test_unsupported_algorithm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            generate_key("NOT_A_REAL_ALGORITHM")

    def test_qkd_failure_uses_fallback_and_marks_source(self):
        with patch(
            "key_manager.generate_qkd_key",
            return_value=("QKD_BB84", None, {"error": "simulator unavailable"}),
        ), patch("key_manager.get_fallback_algorithm", return_value="AES256_GCM"), patch(
            "key_manager.generate_classical_key", return_value="encoded-fallback-key"
        ):
            selected, material, source, metadata = generate_key("QKD_BB84")

        self.assertEqual("AES256_GCM", selected)
        self.assertEqual("encoded-fallback-key", material)
        self.assertEqual("classical", source)
        self.assertIsNone(metadata)


if __name__ == "__main__":
    unittest.main()
