import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from models import DeliveryRequest, DeliveryResponse
from delivery_methods.api_delivery import deliver_via_api
from delivery_methods.mqtt_delivery import deliver_via_mqtt
from delivery_methods.hsm_delivery import deliver_via_hsm
from delivery_methods.file_delivery import deliver_via_file

# Registry of delivery methods
DELIVERY_METHODS = {
    "API": deliver_via_api,
    "MQTT": deliver_via_mqtt,
    "HSM": deliver_via_hsm,
    "FILE": deliver_via_file,
}

# In-memory delivery tracking (in production, use Redis/DB)
delivery_tracker: Dict[str, DeliveryResponse] = {}


async def deliver_key(req: DeliveryRequest) -> DeliveryResponse:
    """
    Main delivery orchestrator.
    Routes the delivery request to the appropriate method handler.
    """
    delivery_id = str(uuid.uuid4())

    print(f"[KDE] Starting delivery {delivery_id} for session {req.session_id}")
    print(f"[KDE] Method: {req.delivery_method}, Destination: {req.destination}")

    try:
        # Get the delivery method handler
        delivery_handler = DELIVERY_METHODS.get(req.delivery_method.upper())

        if not delivery_handler:
            return DeliveryResponse(
                session_id=req.session_id,
                destination=req.destination,
                status="failed",
                delivery_method=req.delivery_method,
                timestamp=datetime.utcnow(),
                delivery_id=delivery_id,
                message=f"Unsupported delivery method: {req.delivery_method}"
            )

        # Execute the delivery
        result = await delivery_handler(req, delivery_id)

        # Track the delivery
        delivery_tracker[delivery_id] = result

        print(f"[KDE] Delivery {delivery_id} completed with status: {result.status}")
        return result

    except Exception as e:
        error_result = DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method=req.delivery_method,
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"Delivery error: {str(e)}"
        )
        delivery_tracker[delivery_id] = error_result
        print(f"[KDE] Delivery {delivery_id} failed: {e}")
        return error_result


def get_delivery_status(delivery_id: str) -> Optional[DeliveryResponse]:
    """Get the status of a specific delivery."""
    return delivery_tracker.get(delivery_id)


def list_deliveries() -> Dict[str, DeliveryResponse]:
    """List all tracked deliveries."""
    return delivery_tracker.copy()