import unittest

from pydantic import ValidationError

from models import CreateKeyRequest


class CreateKeyRequestValidationTests(unittest.TestCase):
    def test_rejects_blank_algorithm(self):
        with self.assertRaises(ValidationError):
            CreateKeyRequest(algorithm="   ")

    def test_rejects_oversized_identifiers(self):
        with self.assertRaises(ValidationError):
            CreateKeyRequest(
                algorithm="AES256_GCM",
                session_id="s" * 257,
            )

        with self.assertRaises(ValidationError):
            CreateKeyRequest(
                algorithm="AES256_GCM",
                request_id="r" * 257,
            )

    def test_trims_identifier_whitespace(self):
        request = CreateKeyRequest(
            algorithm=" AES256_GCM ",
            session_id=" session-1 ",
            request_id=" request-1 ",
        )

        self.assertEqual("AES256_GCM", request.algorithm)
        self.assertEqual("session-1", request.session_id)
        self.assertEqual("request-1", request.request_id)


if __name__ == "__main__":
    unittest.main()