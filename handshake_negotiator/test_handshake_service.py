import unittest
from unittest.mock import patch

from models import NegotiationRequest
from negotiator import negotiate_algorithms


class HandshakeNegotiatorServiceTests(unittest.TestCase):
    """
    Testes de regras de negócio e serviço de negociação (algoritmos, prioridades e fallbacks).
    """

    def test_negotiate_exact_match_highest_priority(self):
        # QKD_BB84 is top priority in policies.yaml
        req = NegotiationRequest(
            source="station-a",
            destination="http://node-b",
            proposed=["AES256_GCM", "QKD_BB84", "Kyber1024"],
            dst_props={"algorithms": ["AES256_GCM", "QKD_BB84"]}
        )
        with patch("negotiator.is_quantum_available", return_value=True):
            chosen, session_id, fallback, reason = negotiate_algorithms(req)

        self.assertEqual("QKD_BB84", chosen)
        self.assertFalse(fallback)
        self.assertIsNone(reason)
        self.assertIsNotNone(session_id)

    def test_negotiate_qkd_fallback_when_quantum_unavailable(self):
        # When QKD is selected but quantum is not available, fallback is Kyber1024
        req = NegotiationRequest(
            source="station-a",
            destination="http://node-b",
            proposed=["QKD_BB84", "AES256_GCM"],
            dst_props={"algorithms": ["QKD_BB84", "AES256_GCM"]}
        )
        with patch("negotiator.is_quantum_available", return_value=False):
            chosen, session_id, fallback, reason = negotiate_algorithms(req)

        self.assertEqual("Kyber1024", chosen)
        self.assertTrue(fallback)
        self.assertEqual("QKD_UNAVAILABLE", reason)

    def test_negotiate_pqc_when_both_support_kyber(self):
        req = NegotiationRequest(
            source="station-a",
            destination="http://node-b",
            proposed=["Kyber768", "AES256_GCM"],
            dst_props={"algorithms": ["Kyber768", "AES256_GCM"]}
        )
        with patch("negotiator.is_quantum_available", return_value=True):
            chosen, session_id, fallback, reason = negotiate_algorithms(req)

        self.assertEqual("Kyber768", chosen)
        self.assertFalse(fallback)
        self.assertIsNone(reason)

    def test_negotiate_classical_algorithm(self):
        req = NegotiationRequest(
            source="station-a",
            destination="http://node-b",
            proposed=["RSA4096", "AES256_GCM"],
            dst_props={"algorithms": ["AES256_GCM"]}
        )
        chosen, session_id, fallback, reason = negotiate_algorithms(req)

        self.assertEqual("AES256_GCM", chosen)
        self.assertFalse(fallback)
        self.assertIsNone(reason)

    def test_negotiate_no_common_algorithm_triggers_default_fallback(self):
        # Default fallback in policies.yaml is ChaCha20_Poly1305
        req = NegotiationRequest(
            source="station-a",
            destination="http://node-b",
            proposed=["UNKNOWN_ALG_1"],
            dst_props={"algorithms": ["UNKNOWN_ALG_2"]}
        )
        chosen, session_id, fallback, reason = negotiate_algorithms(req)

        self.assertEqual("ChaCha20_Poly1305", chosen)
        self.assertTrue(fallback)
        self.assertEqual("NO_COMMON_ALGORITHMS", reason)

    def test_negotiate_without_dst_props_defaults_to_proposed(self):
        req = NegotiationRequest(
            source="station-a",
            destination="http://node-b",
            proposed=["Kyber512", "AES256_GCM"],
            dst_props=None
        )
        chosen, session_id, fallback, reason = negotiate_algorithms(req)
        self.assertEqual("Kyber512", chosen)
        self.assertFalse(fallback)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
