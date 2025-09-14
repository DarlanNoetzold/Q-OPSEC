import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from models import DeliveryRequest, DeliveryResponse
from destination_engine import deliver_key, get_delivery_status, list_deliveries
from config import HOST, PORT, SUPPORTED_METHODS

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
    """
    Entrega o material de chave ao destino especificado usando o método escolhido.

    Métodos suportados:
    - API: HTTP/REST endpoint
    - MQTT: Message broker
    - HSM: Hardware Security Module
    - FILE: Sistema de arquivos seguro
    """
    result = await deliver_key(req)

    if result.status == "failed":
        # Propaga erro HTTP com o detalhe do handler
        raise HTTPException(status_code=500, detail=result.message or "Delivery failed")

    return result


@app.get("/delivery/{delivery_id}", response_model=DeliveryResponse)
async def get_delivery(delivery_id: str):
    """Recupera o status de uma entrega específica."""
    result = get_delivery_status(delivery_id)
    if not result:
        raise HTTPException(status_code=404, detail="Delivery not found")
    return result


@app.get("/deliveries")
async def list_all_deliveries() -> Dict[str, Any]:
    """Lista todas as entregas trackeadas (debug/admin)."""
    deliveries = list_deliveries()
    # FastAPI serializa modelos Pydantic automaticamente
    return {
        "total": len(deliveries),
        "deliveries": deliveries,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "OraculumPrisec KDE",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)