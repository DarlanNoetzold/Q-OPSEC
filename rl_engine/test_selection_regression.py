import unittest
from pathlib import Path

from environment import CryptoAlgorithm, SecurityLevel, map_security_level
from service import ImprovedRLEngineService


class SelectionRegressionTests(unittest.TestCase):
    def test_security_mapping_covers_multiple_levels(self):
        self.assertEqual(SecurityLevel.VERY_LOW, map_security_level(0.05, 0.05))
        self.assertEqual(SecurityLevel.LOW, map_security_level(0.25, 0.20))
        self.assertEqual(SecurityLevel.MODERATE, map_security_level(0.45, 0.40))
        self.assertEqual(SecurityLevel.HIGH, map_security_level(0.70, 0.60))
        self.assertEqual(SecurityLevel.VERY_HIGH, map_security_level(0.85, 0.80))
        self.assertEqual(SecurityLevel.ULTRA, map_security_level(0.95, 0.90))

    def test_algorithm_names_are_compatible_with_handshake_and_kms(self):
        service = ImprovedRLEngineService(registry_path=Path("/tmp/qopsec-test-registry.json"))
        algorithms = service._build_algorithm_list(CryptoAlgorithm.PQC_KYBER, service.env.extract_features({}))
        self.assertIn("Kyber768", algorithms)
        self.assertIn("AES256_GCM", algorithms)
        self.assertNotIn("PQC_KYBER", algorithms)
        self.assertNotIn("AES_256_GCM", algorithms)


if __name__ == "__main__":
    unittest.main()
