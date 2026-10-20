import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment import CryptoAlgorithm, SecurityLevel
from service import ImprovedRLEngineService


class ImprovedRLEngineServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.registry_path = Path(self.temp_directory.name) / "registry.json"
        self.service = ImprovedRLEngineService(registry_path=self.registry_path)

    def test_decision_for_pqc_capable_destination_includes_pqc_wire_name(self):
        result = self.service.decide_algorithms({
            "source": "source-a",
            "security_level": "HIGH",
            "risk_score": 0.85,
            "conf_score": 0.8,
            "dst_props": {"hardware": ["PQC"]},
        })

        self.assertIn("Kyber768", result)
        self.assertIn("AES256_GCM", result)
        self.assertEqual(1, self.service.get_metrics()["total_requests"])

    def test_feedback_updates_learning_metrics_and_episode_persists_table(self):
        self.service.decide_algorithms({
            "source": "source-a",
            "security_level": "HIGH",
            "risk_score": 0.8,
            "conf_score": 0.8,
        })

        self.service.process_feedback("request-1", {
            "success": True,
            "latency": 12.0,
            "resource_usage": 0.4,
        })
        self.service.end_episode()

        self.assertEqual(1, self.service.get_metrics()["successful_negotiations"])
        self.assertEqual(1, self.service.episode_count)
        self.assertTrue(self.registry_path.exists())
        self.assertTrue(self.service.registry.load())

    def test_explicit_quantum_requirement_limits_primary_algorithm_to_qkd(self):
        with patch.object(self.service.policy, "select_action", return_value=0):
            result = self.service.decide_algorithms({
                "security_level": "ULTRA",
                "risk_score": 0.99,
                "conf_score": 0.99,
                "available_resources": 1.0,
                "dst_props": {"hardware": ["QKD", "QUANTUM"]},
            })

        self.assertTrue(result[0].startswith("QKD_"))
        self.assertIn("Kyber768", result)
        self.assertIn("AES256_GCM", result)

    def test_reward_for_selected_action_updates_q_table(self):
        self.service.decide_algorithms({
            "source": "source-a",
            "security_level": "HIGH",
            "risk_score": 0.8,
            "conf_score": 0.8,
        })
        selected_action = self.service.current_episode_experiences[-1]["action"]
        state = self.service.current_episode_experiences[-1]["state"]
        before = self.service.agent.agent.get_q_values(state)[selected_action]
        self.service.process_feedback("request-1", {
            "success": True,
            "latency": 5.0,
            "resource_usage": 0.2,
        })

        after = self.service.agent.get_q_table()[state][selected_action]
        self.assertNotEqual(before, after)
        self.assertEqual(SecurityLevel.HIGH,
                         self.service.current_episode_experiences[-1]["security_level"])

    def test_disabled_training_does_not_record_experiences_or_feedback(self):
        self.service.set_training_mode(False)

        self.service.decide_algorithms({"security_level": "MODERATE"})
        self.service.process_feedback("request-1", {"success": True})

        self.assertEqual([], self.service.current_episode_experiences)
        self.assertEqual(0, self.service.get_metrics()["successful_negotiations"])
        self.assertEqual(0, self.service.get_metrics()["failed_negotiations"])


if __name__ == "__main__":
    unittest.main()
