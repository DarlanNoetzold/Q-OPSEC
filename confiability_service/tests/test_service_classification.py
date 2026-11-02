import unittest
from unittest.mock import patch

from services.conf_model_service import ConfidentialityModelService, synth_corpus
from models.schemas import ClassifyRequest


class DatasetGenerationServiceTests(unittest.TestCase):
    def test_synthetic_corpus_is_reproducible_balanced_and_non_empty(self):
        texts, labels = synth_corpus(3, seed=17)

        self.assertEqual(len(texts), 12)
        self.assertEqual(len(labels), 12)
        self.assertEqual(set(labels), {"public", "internal", "confidential", "restricted"})
        self.assertEqual(texts, synth_corpus(3, seed=17)[0])


class ClassificationServiceTests(unittest.TestCase):
    def setUp(self):
        with patch("services.conf_model_service.read_registry", return_value={"best_conf_model": None}):
            self.service = ConfidentialityModelService()

    @staticmethod
    def request(text, metadata=None):
        return ClassifyRequest.model_validate(
            {
                "request_id": "service-test",
                "content_pointer": {"sample_text": text, "metadata": metadata or {}},
            }
        )

    def test_fallback_classification_escalates_sensitive_terms_and_scans_pii(self):
        result = self.service.classify(self.request("Customer email alice@example.com"))

        self.assertEqual(result.classification, "confidential")
        self.assertIn("email", result.detected_patterns)
        self.assertIn("PII", result.tags)
        self.assertEqual(result.model_version, "conf-fallback-heuristic-1.0")

    def test_empty_input_uses_safe_public_fallback(self):
        result = self.service.classify(self.request("  "))

        self.assertEqual(result.classification, "public")
        self.assertEqual(result.score, 0.3)
        self.assertEqual(result.detected_patterns, [])

    def test_loaded_model_output_is_escalated_and_metadata_is_preserved(self):
        class DeterministicModel:
            classes_ = ["public", "internal", "confidential", "restricted"]

            def predict_proba(self, texts):
                return [[0.8, 0.1, 0.05, 0.05]]

        self.service._best_model = DeterministicModel()
        self.service._best_info = {"name": "test-model"}
        result = self.service.classify(
            self.request("alice@example.com", {"doc_type": "hr", "app": "portal", "user_label": "staff"})
        )

        self.assertEqual(result.classification, "confidential")
        self.assertEqual(result.score, 0.93)
        self.assertEqual(result.model_version, "test-model")
        self.assertEqual(result.source_app_context, "portal")
        self.assertEqual(result.user_label, "staff")
        self.assertEqual(result.tags, ["PII", "hr"])

    def test_unavailable_model_returns_none_for_http_layer_to_report(self):
        self.assertIsNone(self.service.classify(self.request("ordinary public announcement")))


if __name__ == "__main__":
    unittest.main()
