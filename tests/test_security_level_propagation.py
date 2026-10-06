import json
import unittest
from pathlib import Path

from orchestrator_linux import resolve_rl_security_level


class ScenarioCoverageTests(unittest.TestCase):
    def test_scenarios_cover_all_rl_levels(self):
        scenarios = json.loads(
            (Path(__file__).parents[1] / "tests" / "pipeline_scenarios.json").read_text()
        )["scenarios"]
        levels = {s.get("security_level") for s in scenarios}
        self.assertEqual(
            {"VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "ULTRA"},
            levels,
        )

    def test_explicit_levels_are_preserved(self):
        for level in ("VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "ULTRA"):
            self.assertEqual(level, resolve_rl_security_level(level, 0.01))

    def test_scenario_payload_carries_level_and_algorithms(self):
        scenarios = json.loads(
            (Path(__file__).parents[1] / "tests" / "pipeline_scenarios.json").read_text()
        )["scenarios"]
        moderate = next(s for s in scenarios if s["id"] == "scenario-01-baseline")
        self.assertEqual("MODERATE", moderate["payload"]["security_level"])
        self.assertEqual(moderate["proposed_algorithms"], moderate["payload"]["proposed"])

    def test_extended_experiment_matrix_is_present(self):
        scenarios = json.loads(
            (Path(__file__).parents[1] / "tests" / "pipeline_scenarios.json").read_text()
        )["scenarios"]
        ids = {scenario["id"] for scenario in scenarios}
        self.assertTrue({
            "scenario-data-exfiltration",
            "scenario-replay-attack",
            "scenario-zero-trust-device",
            "scenario-pqc-migration",
            "scenario-degraded-dependencies",
            "scenario-malformed-payload",
        }.issubset(ids))
        self.assertGreaterEqual(len(scenarios), 20)

    def test_additional_experiment_matrix_is_present(self):
        scenarios = json.loads(
            (Path(__file__).parents[1] / "tests" / "pipeline_scenarios.json").read_text()
        )["scenarios"]
        ids = {scenario["id"] for scenario in scenarios}
        self.assertTrue({
            "scenario-key-rotation",
            "scenario-cross-tenant-isolation",
            "scenario-large-payload-boundary",
            "scenario-clock-skew",
            "scenario-duplicate-delivery",
            "scenario-unicode-payload",
            "scenario-policy-conflict",
            "scenario-audit-integrity",
            "scenario-key-compromise",
            "scenario-algorithm-downgrade",
            "scenario-rate-limit-burst",
            "scenario-network-partition",
            "scenario-privacy-redaction",
            "scenario-invalid-algorithm",
            "scenario-concurrent-key-use",
            "scenario-policy-version-skew",
        }.issubset(ids))
        for scenario in scenarios:
            self.assertIn("security_level", scenario)
            self.assertIn("proposed_algorithms", scenario)
            self.assertIn("payload", scenario)
            self.assertIn("source", scenario["payload"])

    def test_new_orchestrator_experiments_cover_additional_threats(self):
        scenarios = json.loads(
            (Path(__file__).parents[1] / "tests" / "pipeline_scenarios.json").read_text()
        )["scenarios"]
        ids = {scenario["id"] for scenario in scenarios}
        self.assertTrue({
            "scenario-mfa-fatigue",
            "scenario-supply-chain-compromise",
            "scenario-container-escape-attempt",
            "scenario-secrets-injection",
            "scenario-malware-ransomware-indicator",
            "scenario-dns-tunneling",
            "scenario-api-schema-abuse",
            "scenario-authorization-boundary",
            "scenario-crypto-key-expiry",
            "scenario-resource-exhaustion",
        }.issubset(ids))


if __name__ == "__main__":
    unittest.main()
