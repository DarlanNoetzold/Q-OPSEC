import httpx
from datetime import datetime
from models import DeliveryRequest, DeliveryResponse
from config import API_TIMEOUT


async def deliver_via_api(req: DeliveryRequest, delivery_id: str) -> DeliveryResponse:
    try:
        payload = {
            "session_id": req.session_id,
            "algorithm": req.algorithm,
            "key_material": req.key_material,
            "expires_at": req.expires_at.isoformat(),
            "delivery_id": delivery_id
        }
        if req.metadata:
            payload["metadata"] = req.metadata

        url = f"{req.destination}/receive_key" if req.destination.startswith("http") \
              else f"http://{req.destination}/receive_key"

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.post(url, json=payload)

        if 200 <= resp.status_code < 300:
            return DeliveryResponse(
                session_id=req.session_id,
                destination=req.destination,
                status="delivered",
                delivery_method="API",
                timestamp=datetime.utcnow(),
                delivery_id=delivery_id,
                message=f"Key delivered successfully to {url}",
                metadata={"http_status": resp.status_code, "response": resp.text[:500]}
            )
        else:
            return DeliveryResponse(
                session_id=req.session_id,
                destination=req.destination,
                status="failed",
                delivery_method="API",
                timestamp=datetime.utcnow(),
                delivery_id=delivery_id,
                message=f"HTTP {resp.status_code}: {resp.text[:500]}",
                metadata={"url": url}
            )

    except httpx.RequestError as e:
        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method="API",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"Request error: {e.__class__.__name__}: {e}",
            metadata={"url": url}
        )
    except Exception as e:
        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method="API",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"API delivery failed: {e}",
            metadata={"url": url}
        )