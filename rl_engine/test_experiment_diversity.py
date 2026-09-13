import json
import unittest
from pathlib import Path

from environment import CryptoAlgorithm, SecurityLevel
from service import ImprovedRLEngineService


class ExperimentDiversityTests(unittest.TestCase):
    def setUp(self):
        self.service = ImprovedRLEngineService(
            registry_path=Path("/tmp/qopsec-diversity-test-registry.json")
        )
        self.service.set_training_mode(False)

    def test_selected_algorithms_use_wire_names(self):
        algorithms = self.service._build_algorithm_list(
            CryptoAlgorithm.PQC_KYBER,
            self.service.env.extract_features({}),
        )
        self.assertEqual("Kyber768", algorithms[0])
        self.assertIn("AES256_GCM", algorithms)
        self.assertNotIn("PQC_KYBER", algorithms)
        self.assertNotIn("AES_256_GCM", algorithms)

    def test_high_and_ultra_contexts_prefer_pqc_or_qkd(self):
        high = self.service.decide_algorithms({
            "source": "high-node",
            "security_level": "HIGH",
            "risk_score": 0.85,
            "conf_score": 0.90,
            "dst_props": {"hardware": ["PQC"]},
        })
        ultra = self.service.decide_algorithms({
            "source": "quantum-node",
            "security_level": "ULTRA",
            "risk_score": 0.95,
            "conf_score": 0.98,
            "dst_props": {"hardware": ["QKD", "QUANTUM"]},
        })
        high_results = [self.service.decide_algorithms({
            "source": "high-node",
            "security_level": "HIGH",
            "risk_score": 0.85,
            "conf_score": 0.90,
            "dst_props": {"hardware": ["PQC"]},
        }) for _ in range(30)]
        ultra_results = [self.service.decide_algorithms({
            "source": "quantum-node",
            "security_level": "ULTRA",
            "risk_score": 0.95,
            "conf_score": 0.98,
            "dst_props": {"hardware": ["QKD", "QUANTUM"]},
        }) for _ in range(30)]
        self.assertTrue(any(a.startswith(("Kyber", "ML-KEM", "NTRU", "QKD"))
                           for result in high_results for a in result))
        self.assertTrue(any(a.startswith("QKD") for result in ultra_results for a in result))

    def test_dashboard_scenarios_cover_multiple_levels_and_pqc(self):
        data = json.loads((Path(__file__).parents[1] / "tests/pipeline_scenarios.json").read_text())
        scenarios = data["scenarios"]
        levels = {s.get("security_level") or s.get("payload", {}).get("security_level")
                  for s in scenarios}
        serialized = json.dumps(scenarios)
        self.assertGreaterEqual(len(levels - {None}), 4)
        self.assertTrue(any(token in serialized for token in ("Kyber", "ML-KEM", "Dilithium", "Falcon")))


if __name__ == "__main__":
    unittest.main()
