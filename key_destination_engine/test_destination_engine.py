import asyncio
import unittest

from destination_engine import deliver_key, get_delivery_status, list_deliveries
from models import DeliveryRequest


class DestinationEngineRegressionTests(unittest.TestCase):
    def test_optional_request_id_is_normalized_in_response(self):
        request = DeliveryRequest(
            session_id="session-1",
            destination="test-destination",
            delivery_method="unsupported",
            key_material="test-key",
            algorithm="AES256_GCM",
            expires_at=4102444800,
        )

        result = asyncio.run(deliver_key(request))

        self.assertEqual("req-unknown", result.request_id)
        self.assertEqual(result, get_delivery_status(result.delivery_id))
        self.assertIn(result.delivery_id, list_deliveries())


if __name__ == "__main__":
    unittest.main()