import asyncio
import os
import tempfile
import unittest

from destination_engine import deliver_key, get_delivery_status, list_deliveries
from models import DeliveryRequest
from delivery_methods.mqtt_delivery import deliver_via_mqtt


class DestinationEngineRegressionTests(unittest.TestCase):
    def test_mqtt_delivery_accepts_unix_timestamp_and_preserves_request_id(self):
        async def run_test():
            request = DeliveryRequest(
                session_id="session-mqtt",
                request_id="request-mqtt",
                destination="device-1",
                delivery_method="MQTT",
                key_material="secret-key",
                algorithm="AES256_GCM",
                expires_at=4102444800,
            )
            return await deliver_via_mqtt(request, "delivery-mqtt")

        result = asyncio.run(run_test())

        self.assertEqual("delivered", result.status)
        self.assertEqual("request-mqtt", result.request_id)
        self.assertEqual(4102444800, result.metadata["expires_at"])

    def test_file_delivery_accepts_filename_without_parent_directory(self):
        async def run_test():
            request = DeliveryRequest(
                session_id="session-file",
                request_id="request-file",
                destination="key-material.test",
                delivery_method="FILE",
                key_material="secret-key",
                algorithm="AES256_GCM",
                expires_at=4102444800,
            )
            return await deliver_key(request)

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                result = asyncio.run(run_test())
                self.assertEqual("delivered", result.status)
                with open("key-material.test", encoding="utf-8") as key_file:
                    self.assertEqual("secret-key", key_file.read())
            finally:
                os.chdir(previous_directory)

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