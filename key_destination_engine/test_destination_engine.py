import asyncio
import os
import tempfile
import unittest

from destination_engine import deliver_key, get_delivery_status, list_deliveries, queue_delivery
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

    def test_delivery_request_rejects_blank_required_text(self):
        for field in (
            "session_id",
            "destination",
            "delivery_method",
            "key_material",
            "algorithm",
        ):
            payload = {
                "session_id": "session-1",
                "destination": "device-1",
                "delivery_method": "FILE",
                "key_material": "secret-key",
                "algorithm": "AES256_GCM",
                "expires_at": 4102444800,
            }
            payload[field] = "   "
            with self.subTest(field=field), self.assertRaises(ValueError):
                DeliveryRequest(**payload)

    def test_delivery_request_rejects_blank_request_id(self):
        with self.assertRaises(ValueError):
            DeliveryRequest(
                session_id="session-1",
                request_id="   ",
                destination="device-1",
                delivery_method="FILE",
                key_material="secret-key",
                algorithm="AES256_GCM",
                expires_at=4102444800,
            )


    def test_queued_delivery_returns_immediately_and_completes_in_background(self):
        async def run_test():
            request = DeliveryRequest(
                session_id="session-async",
                request_id="request-async",
                destination="async-key.test",
                delivery_method="FILE",
                key_material="async-secret",
                algorithm="AES256_GCM",
                expires_at=4102444800,
            )
            queued = queue_delivery(request)
            self.assertEqual("queued", queued.status)
            self.assertEqual(queued, get_delivery_status(queued.delivery_id))

            for _ in range(100):
                current = get_delivery_status(queued.delivery_id)
                if current.status != "queued":
                    return current
                await asyncio.sleep(0.01)
            return get_delivery_status(queued.delivery_id)

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                result = asyncio.run(run_test())
                self.assertEqual("delivered", result.status)
                with open("async-key.test", encoding="utf-8") as key_file:
                    self.assertEqual("async-secret", key_file.read())
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()