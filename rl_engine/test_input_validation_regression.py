import unittest

from main import ContextRequest, FeedbackRequest


class RLInputValidationRegressionTests(unittest.TestCase):
    def test_context_rejects_blank_endpoints(self):
        payload = {
            "source": "source-1",
            "destination": "destination-1",
            "security_level": "HIGH",
        }
        for field in ("source", "destination"):
            invalid = dict(payload, **{field: "   "})
            with self.subTest(field=field), self.assertRaises(ValueError):
                ContextRequest(**invalid)

    def test_feedback_rejects_blank_request_id(self):
        with self.assertRaises(ValueError):
            FeedbackRequest(request_id="   ", success=True)


if __name__ == "__main__":
    unittest.main()
