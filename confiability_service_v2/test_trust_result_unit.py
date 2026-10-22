import unittest
from datetime import datetime, timezone

from core.trust_context import TrustContext
from core.trust_result import SignalResult, TrustResult


class TrustResultSerializationTests(unittest.TestCase):
    def test_to_dict_serializes_rounded_scores_and_context_metadata(self):
        context = TrustContext(
            payload={"event": "login"},
            metadata={"entity_id": "user-7", "source_id": "auth"},
        )
        result = TrustResult(
            trust_score=0.87654,
            trust_level="HIGH",
            confidence_interval=(0.81234, 0.92345),
            dimensions={"source": 0.87654},
            risk_flags=["REVIEW"],
            explainability={"summary": "consistent"},
            trust_dna_value="T-DNA-v2:abc12345",
            context=context,
            signal_results=[
                SignalResult("source_reliability", 0.87654, 0.93456, {"history": 4})
            ],
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        actual = result.to_dict()

        self.assertEqual(0.8765, actual["trust_score"])
        self.assertEqual({"lower": 0.8123, "upper": 0.9234}, actual["confidence_interval"])
        self.assertEqual({"source": 0.8765}, actual["dimensions"])
        self.assertEqual("user-7", actual["metadata"]["entity_id"])
        self.assertEqual("auth", actual["metadata"]["source_id"])
        self.assertEqual(1, actual["metadata"]["num_signals"])
        self.assertEqual(0.8765, actual["signals"][0]["score"])
        self.assertEqual(0.9346, actual["signals"][0]["confidence"])


if __name__ == "__main__":
    unittest.main()
