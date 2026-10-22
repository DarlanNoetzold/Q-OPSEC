import unittest

from bootstrap_v2 import reset_trust_engine
from config.trust_config import TrustConfig
from core.orchestrator import TrustOrchestrator
from core.trust_context import TrustContext
from storage.fingerprint_repo import FingerprintRepository
from storage.trust_repository import TrustRepository


class TrustOrchestratorPersistenceTests(unittest.TestCase):
    def setUp(self):
        reset_trust_engine()
        self.trust_repo = TrustRepository()
        self.fingerprint_repo = FingerprintRepository()
        self.orchestrator = TrustOrchestrator(
            config=TrustConfig(),
            trust_repo=self.trust_repo,
            fingerprint_repo=self.fingerprint_repo,
        )

    def tearDown(self):
        reset_trust_engine()

    def test_evaluation_persists_entity_source_and_payload_fingerprint(self):
        context = TrustContext(
            payload={"event": "login", "result": "success"},
            metadata={
                "entity_id": "user-42",
                "source_id": "auth-service",
                "timestamp": "2025-01-01T12:00:00Z",
                "data_type": "event",
            },
        )

        result = self.orchestrator.evaluate(context)

        entity_history = self.trust_repo.get_entity_history("user-42")
        source_history = self.trust_repo.get_source_history("auth-service")
        fingerprint_info = self.fingerprint_repo.get_fingerprint_info(context.payload_fp)

        self.assertEqual(1, len(entity_history))
        self.assertEqual(result.trust_score, entity_history[0]["trust_score"])
        self.assertEqual(1, len(source_history))
        self.assertEqual("user-42", source_history[0]["entity_id"])
        self.assertIsNotNone(fingerprint_info)
        self.assertIn("auth-service", fingerprint_info["sources"])


if __name__ == "__main__":
    unittest.main()
