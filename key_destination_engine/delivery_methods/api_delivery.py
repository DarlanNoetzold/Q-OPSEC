import httpx
from datetime import datetime
from models import DeliveryRequest, DeliveryResponse
from config import API_TIMEOUT


async def deliver_via_api(req: DeliveryRequest, delivery_id: str) -> DeliveryResponse:
    """
    Deliver key material via HTTP/REST API.
    """
    try:
        # Prepare the payload
        payload = {
            "session_id": req.session_id,
            "algorithm": req.algorithm,
            "key_material": req.key_material,
            "expires_at": req.expires_at.isoformat(),
            "delivery_id": delivery_id
        }

        # Add metadata if present
        if req.metadata:
            payload["metadata"] = req.metadata

        # Determine the target URL
        if req.destination.startswith("http"):
            url = f"{req.destination}/receive_key"
        else:
            url = f"http://{req.destination}/receive_key"

        print(f"[KDE] Delivering to API endpoint: {url}")

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code == 200:
            return DeliveryResponse(
                session_id=req.session_id,
                destination=req.destination,
                status="delivered",
                delivery_method="API",
                timestamp=datetime.utcnow(),
                delivery_id=delivery_id,
                message=f"Key delivered successfully to {url}",
                metadata={"http_status": resp.status_code, "response": resp.text[:200]}
            )
        else:
            return DeliveryResponse(
                session_id=req.session_id,
                destination=req.destination,
                status="failed",
                delivery_method="API",
                timestamp=datetime.utcnow(),
                delivery_id=delivery_id,
                message=f"HTTP Error {resp.status_code}: {resp.text[:200]}"
            )

    except Exception as e:
        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method="API",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"API delivery failed: {str(e)}"
        )