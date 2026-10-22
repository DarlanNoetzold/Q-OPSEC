import unittest

from config.trust_config import TrustConfig
from core.trust_context import TrustContext
from signals.anomaly import AnomalyDetectionSignal
from signals.source import SourceReliabilitySignal
from storage.trust_repository import TrustRepository


class TrustSignalsUnitTests(unittest.TestCase):
    def setUp(self):
        self.config = TrustConfig()
        self.repository = TrustRepository()

    def test_source_reliability_returns_neutral_low_confidence_without_source(self):
        signal = SourceReliabilitySignal(self.config, self.repository)
        context = TrustContext(payload={"event": "login"}, metadata={})

        result = signal.evaluate(context)

        self.assertEqual("source_reliability", result.signal_name)
        self.assertEqual(0.5, result.score)
        self.assertEqual(0.3, result.confidence)
        self.assertEqual("no_source_id", result.metadata["reason"])

    def test_anomaly_signal_flags_tiny_payload(self):
        signal = AnomalyDetectionSignal(self.config, self.repository)
        context = TrustContext(payload={"x": 1}, metadata={})

        result = signal.evaluate(context)

        self.assertEqual(0.7, result.score)
        self.assertTrue(result.metadata["is_anomaly"])
        self.assertIn("tiny_payload", result.metadata["anomalies"])


if __name__ == "__main__":
    unittest.main()
