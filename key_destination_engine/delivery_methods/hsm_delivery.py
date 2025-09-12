from datetime import datetime
from models import DeliveryRequest, DeliveryResponse


async def deliver_via_hsm(req: DeliveryRequest, delivery_id: str) -> DeliveryResponse:
    """
    Deliver key material to Hardware Security Module (HSM).
    """
    try:
        # Placeholder for HSM integration
        # In production, use PyKCS11 or similar

        print(f"[KDE] Loading key into HSM slot for destination: {req.destination}")

        # Simulate HSM operation
        import asyncio
        await asyncio.sleep(0.2)  # Simulate HSM processing time

        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="delivered",
            delivery_method="HSM",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message="Key securely loaded into HSM",
            metadata={"hsm_slot": req.destination, "key_handle": f"handle_{delivery_id[:8]}"}
        )

    except Exception as e:
        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method="HSM",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"HSM delivery failed: {str(e)}"
        )