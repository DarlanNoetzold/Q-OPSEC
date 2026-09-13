import json
import unittest
from pathlib import Path

from service import ImprovedRLEngineService


class PipelinePropagationRegressionTests(unittest.TestCase):
    def test_scenarios_expose_explicit_security_levels(self):
        scenarios = json.loads(
            (Path(__file__).parents[1] / "tests/pipeline_scenarios.json").read_text()
        )["scenarios"]
        levels = {scenario.get("security_level") for scenario in scenarios}
        self.assertGreaterEqual(len(levels), 5)
        self.assertNotEqual({"LOW"}, levels)

    def test_explicit_level_is_not_lost_by_rl_selection(self):
        service = ImprovedRLEngineService(
            registry_path=Path("/tmp/qopsec-level-regression.json")
        )
        context = {
            "source": "high-node",
            "security_level": "HIGH",
            "risk_score": 0.85,
            "conf_score": 0.90,
            "dst_props": {"hardware": ["PQC"]},
        }
        self.assertEqual("HIGH", service._get_security_level_name(context))
        self.assertTrue(any(a.startswith(("Kyber", "Dilithium", "Falcon", "NTRU", "Saber"))
                           for a in service.decide_algorithms(context)))


if __name__ == "__main__":
    unittest.main()
