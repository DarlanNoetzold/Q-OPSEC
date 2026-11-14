import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from models import DeliveryRequest, DeliveryResponse
from destination_engine import deliver_key, queue_delivery, get_delivery_status, list_deliveries
from config import HOST, PORT, SUPPORTED_METHODS
from common.observability import instrument_app

logger = logging.getLogger("key_destination_engine")

app = FastAPI(
    title="OraculumPrisec Key Destination Engine",
    description="Secure key delivery system supporting multiple transport methods",
    version="1.1.0",
)


@app.get("/")
async def root():
    return {
        "service": "OraculumPrisec Key Destination Engine",
        "version": "1.1.0",
        "status": "operational",
        "supported_methods": SUPPORTED_METHODS,
    }


@app.post("/deliver", response_model=DeliveryResponse)
async def deliver(req: DeliveryRequest):
    logger.info("[%s] KDE: Delivering key to %s", req.request_id or "req-unknown", req.destination)
    result = await deliver_key(req)

    if result.status == "failed":
        raise HTTPException(status_code=500, detail=result.message or "Delivery failed")

    return result


@app.post("/deliver/async", response_model=DeliveryResponse, status_code=202)
async def deliver_async(req: DeliveryRequest):
    """Accept the delivery and return immediately; poll /delivery/{delivery_id} for the outcome."""
    logger.info("[%s] KDE: Queued delivery to %s", req.request_id or "req-unknown", req.destination)
    return queue_delivery(req)


@app.get("/delivery/{delivery_id}", response_model=DeliveryResponse)
async def get_delivery(delivery_id: str):
    result = get_delivery_status(delivery_id)
    if not result:
        raise HTTPException(status_code=404, detail="Delivery not found")
    return result


@app.get("/deliveries")
async def list_all_deliveries() -> Dict[str, Any]:
    deliveries = list_deliveries()
    return {
        "total": len(deliveries),
        "deliveries": deliveries,
    }


@app.on_event("shutdown")
async def _shutdown() -> None:
    from delivery_methods.api_delivery import close_http_client

    await close_http_client()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "OraculumPrisec KDE",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)


instrument_app(app)
