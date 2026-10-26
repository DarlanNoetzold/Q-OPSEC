import unittest
from unittest.mock import patch

from models import NegotiationRequest, NegotiationResponse
from negotiator import load_policies, is_quantum_available, negotiate_algorithms, create_session
from main import normalize_destination


class HandshakeNegotiatorUnitTests(unittest.TestCase):
    """
    Testes unitários puros para modelos, validações, regras de negócio e helpers.
    """

    def test_negotiation_request_valid(self):
        req = NegotiationRequest(
            request_id="req-123",
            source="client-a",
            destination="http://192.168.18.18:8080",
            proposed=["Kyber1024", "AES256_GCM"],
            dst_props={"algorithms": ["Kyber1024", "AES256_GCM"]}
        )
        self.assertEqual("req-123", req.request_id)
        self.assertEqual("client-a", req.source)
        self.assertEqual("http://192.168.18.18:8080", req.destination)
        self.assertEqual(["Kyber1024", "AES256_GCM"], req.proposed)

    def test_negotiation_request_rejects_empty_or_whitespace_source_and_destination(self):
        with self.assertRaises(ValueError):
            NegotiationRequest(
                source="   ",
                destination="http://example.com",
                proposed=["AES256_GCM"]
            )
        with self.assertRaises(ValueError):
            NegotiationRequest(
                source="client-a",
                destination="",
                proposed=["AES256_GCM"]
            )

    def test_negotiation_request_rejects_empty_proposed_list(self):
        with self.assertRaises(ValueError):
            NegotiationRequest(
                source="client-a",
                destination="http://example.com",
                proposed=[]
            )

    def test_negotiation_request_rejects_blank_elements_in_proposed(self):
        with self.assertRaises(ValueError):
            NegotiationRequest(
                source="client-a",
                destination="http://example.com",
                proposed=["AES256_GCM", "   "]
            )

    def test_load_policies(self):
        policies = load_policies("policies.yaml")
        self.assertIn("priority", policies)
        self.assertIn("fallback", policies)
        self.assertIn("QKD_BB84", policies["priority"])
        self.assertIn("Kyber1024", policies["priority"])

    def test_is_quantum_available_mockable(self):
        with patch("negotiator.random.random", return_value=0.5):
            self.assertTrue(is_quantum_available())
        with patch("negotiator.random.random", return_value=0.3):
            self.assertFalse(is_quantum_available())

    def test_create_session(self):
        session_id, expires_at = create_session("Kyber1024", ttl_seconds=60)
        self.assertIsNotNone(session_id)
        self.assertIsNotNone(expires_at)

    def test_normalize_destination_variations(self):
        # Empty or root path appends /receiver
        self.assertEqual("http://192.168.18.18:8080/receiver", normalize_destination("http://192.168.18.18:8080"))
        self.assertEqual("http://192.168.18.18:8080/receiver", normalize_destination("http://192.168.18.18:8080/"))
        
        # Existing subpath is preserved
        self.assertEqual("http://192.168.18.18:8080/custom/path", normalize_destination("http://192.168.18.18:8080/custom/path"))
        
        # Non-http/https is returned as-is
        self.assertEqual("tcp://192.168.18.18:9000", normalize_destination("tcp://192.168.18.18:9000"))
        self.assertEqual("invalid-uri", normalize_destination("invalid-uri"))


if __name__ == "__main__":
    unittest.main()
