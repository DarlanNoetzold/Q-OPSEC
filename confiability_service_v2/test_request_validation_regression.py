import unittest

from api.v2.schemas import TrustEvaluationRequest


class TrustRequestValidationRegressionTests(unittest.TestCase):
    def test_evaluation_rejects_empty_payload(self):
        with self.assertRaises(ValueError):
            TrustEvaluationRequest(payload={})

    def test_evaluation_keeps_metadata_optional(self):
        request = TrustEvaluationRequest(payload={"event": "login"})
        self.assertEqual({}, request.metadata)


if __name__ == "__main__":
    unittest.main()
