import tempfile
import unittest
from pathlib import Path

from environment import (
    CryptoAlgorithm,
    EnhancedEnvironment,
    SecurityLevel,
    map_security_level,
)
from policies import EpsilonGreedyPolicy, UCBPolicy
from registry import RLRegistry


class EnvironmentUnitTests(unittest.TestCase):
    def setUp(self):
        self.environment = EnhancedEnvironment()

    def test_feature_extraction_uses_safe_defaults_and_detects_capabilities(self):
        features = self.environment.extract_features({
            "source": "edge-a",
            "risk_score": None,
            "dst_props": {"hardware": ["qkd", "quantum"]},
        })

        self.assertEqual("edge-a", features.source_device_type)
        self.assertEqual(0.5, features.current_threat_level)
        self.assertTrue(features.qkd_available)
        self.assertTrue(features.quantum_hardware_present)

    def test_invalid_quantum_action_is_filtered_without_capability(self):
        features = self.environment.extract_features({"available_resources": 1.0})

        valid_actions = self.environment.get_valid_actions(features, SecurityLevel.ULTRA)

        self.assertNotIn(CryptoAlgorithm.QKD_BB84, valid_actions)
        self.assertTrue(valid_actions)

    def test_security_mapping_uses_the_most_pessimistic_signal(self):
        self.assertEqual(SecurityLevel.ULTRA, map_security_level(0.95, 0.1))

    def test_successful_pqc_action_receives_positive_reward(self):
        features = self.environment.extract_features({"available_resources": 1.0})

        reward = self.environment.compute_reward(
            CryptoAlgorithm.PQC_KYBER,
            features,
            SecurityLevel.HIGH,
            {"success": True, "latency": 10, "resource_usage": 0.2},
        )

        self.assertGreater(reward, 0)


class PolicyUnitTests(unittest.TestCase):
    def test_epsilon_greedy_exploits_highest_q_value_when_exploration_is_disabled(self):
        policy = EpsilonGreedyPolicy(epsilon=0)

        selected = policy.select_action("state", {1: 3.0, 2: 9.0}, [1, 2])

        self.assertEqual(2, selected)

    def test_ucb_selects_each_untried_valid_action_before_repeating(self):
        policy = UCBPolicy()

        first = policy.select_action("state", {}, [4, 7])
        second = policy.select_action("state", {}, [4, 7])

        self.assertEqual({4, 7}, {first, second})


class RegistryUnitTests(unittest.TestCase):
    def test_q_table_round_trips_through_json(self):
        q_table = {"R8|C7": {"2": 1.25, "3": -0.5}}
        with tempfile.TemporaryDirectory() as directory:
            registry = RLRegistry(Path(directory) / "registry.json")

            registry.save(q_table)

            self.assertEqual(q_table, registry.load())

    def test_missing_registry_loads_as_empty_table(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = RLRegistry(Path(directory) / "missing.json")

            self.assertEqual({}, registry.load())


if __name__ == "__main__":
    unittest.main()
