import unittest
from unittest.mock import patch

from models import NegotiationRequest
from negotiator import negotiate_algorithms


class NegotiationRegressionTests(unittest.TestCase):
    def test_pqc_proposal_is_selected_when_supported(self):
        request = NegotiationRequest(
            request_id="test-pqc",
            source="secure-source",
            destination="secure-destination",
            proposed=["Kyber768", "AES256_GCM"],
            dst_props={"algorithms": ["Kyber768", "AES256_GCM"]},
        )
        with patch("negotiator.is_quantum_available", return_value=True):
            selected, _, fallback, reason = negotiate_algorithms(request)

        self.assertEqual("Kyber768", selected)
        self.assertFalse(fallback)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
