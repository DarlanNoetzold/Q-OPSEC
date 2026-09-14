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


if __name__ == "__main__":
    unittest.main()
