import unittest

from models.schemas import ClassifyRequest, TrainRequest, validate_payload
from repositories.patterns_repo import PatternsRepository


class SchemaUnitTests(unittest.TestCase):
    def test_classify_request_accepts_optional_context_and_metadata(self):
        request, error = validate_payload(
            ClassifyRequest,
            {
                "request_id": "unit-1",
                "content_pointer": {
                    "sample_text": "internal roadmap",
                    "metadata": {"doc_type": "hr", "app": "portal"},
                },
            },
        )

        self.assertIsNone(error)
        self.assertEqual(request.content_pointer.metadata["doc_type"], "hr")
        self.assertIsNone(request.source)

    def test_classify_request_rejects_missing_required_fields(self):
        request, error = validate_payload(ClassifyRequest, {"request_id": "unit-2"})

        self.assertIsNone(request)
        self.assertIsNotNone(error)
        self.assertIn(("content_pointer",), {tuple(item["loc"]) for item in error.errors()})

    def test_train_request_enforces_sample_bounds(self):
        _, error = validate_payload(TrainRequest, {"n_per_class": 19})
        self.assertIsNotNone(error)

        request, error = validate_payload(TrainRequest, {"n_per_class": 20})
        self.assertIsNone(error)
        self.assertEqual(request.n_per_class, 20)


class PatternRepositoryUnitTests(unittest.TestCase):
    def setUp(self):
        self.patterns = PatternsRepository()

    def test_scan_detects_supported_pii_patterns_without_returning_values(self):
        findings = self.patterns.scan(
            "Contact alice@example.com, ID 123.456.789-00 and card 4111111111111111"
        )

        self.assertEqual(
            {item["type"]: item["count"] for item in findings},
            {"credit_card": 1, "email": 1, "national_id": 1},
        )
        self.assertNotIn("alice@example.com", str(findings))

    def test_tags_are_deduplicated_and_empty_text_is_safe(self):
        findings = self.patterns.scan(None)
        self.assertEqual(findings, [])
        self.assertEqual(self.patterns.tags_from_findings(findings), [])
        self.assertEqual(
            self.patterns.tags_from_findings(
                [{"type": "email"}, {"type": "credit_card"}, {"type": "email"}]
            ),
            ["PII", "finance"],
        )


if __name__ == "__main__":
    unittest.main()
