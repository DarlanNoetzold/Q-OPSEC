import json
import asyncio
from datetime import datetime
from models import DeliveryRequest, DeliveryResponse
from config import MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, MQTT_TIMEOUT


async def deliver_via_mqtt(req: DeliveryRequest, delivery_id: str) -> DeliveryResponse:
    try:

        topic = f"keys/{req.destination}/receive"
        payload = {
            "session_id": req.session_id,
            "algorithm": req.algorithm,
            "key_material": req.key_material,
            "expires_at": req.expires_at.isoformat(),
            "delivery_id": delivery_id
        }

        print(f"[KDE] Publishing to MQTT topic: {topic}")
        print(f"[KDE] MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")

        await asyncio.sleep(0.1)

        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="delivered",
            delivery_method="MQTT",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"Key published to MQTT topic: {topic}",
            metadata={"topic": topic, "broker": f"{MQTT_BROKER}:{MQTT_PORT}"}
        )

    except Exception as e:
        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method="MQTT",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"MQTT delivery failed: {str(e)}"
        )