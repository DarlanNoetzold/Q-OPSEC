import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union

from models import DeliveryRequest, DeliveryResponse
from config import SUPPORTED_METHODS

# As implementações reais devem existir em delivery_methods/*
# Mantemos a assinatura esperada: (req: DeliveryRequest, delivery_id: str) -> DeliveryResponse
# Se algum handler retornar tupla, nós normalizamos abaixo.
try:
    from delivery_methods.api_delivery import deliver_via_api    # type: ignore
except Exception:  # fallback opcional para evitar crash se módulo não existir em dev
    deliver_via_api = None

try:
    from delivery_methods.mqtt_delivery import deliver_via_mqtt  # type: ignore
except Exception:
    deliver_via_mqtt = None

try:
    from delivery_methods.hsm_delivery import deliver_via_hsm    # type: ignore
except Exception:
    deliver_via_hsm = None

try:
    from delivery_methods.file_delivery import deliver_via_file  # type: ignore
except Exception:
    deliver_via_file = None

# Registry de métodos
DELIVERY_METHODS: Dict[str, Optional[object]] = {
    "API": deliver_via_api,
    "MQTT": deliver_via_mqtt,
    "HSM": deliver_via_hsm,
    "FILE": deliver_via_file,
}

# Tracking in-memory (em produção, use Redis/DB)
delivery_tracker: Dict[str, DeliveryResponse] = {}
delivery_attempts: Dict[str, int] = {}


def _build_response(
    req: DeliveryRequest,
    delivery_id: str,
    status: str,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DeliveryResponse:
    return DeliveryResponse(
        session_id=req.session_id,
        request_id=req.request_id,
        destination=req.destination,
        status=status,
        delivery_method=req.delivery_method,
        timestamp=datetime.utcnow(),
        delivery_id=delivery_id,
        message=message,
        metadata=metadata,
    )


def _normalize_handler_result(
    req: DeliveryRequest,
    delivery_id: str,
    result: Union[
        DeliveryResponse,
        Tuple[str, str],
        Tuple[str, str, Optional[Dict[str, Any]]],
    ],
) -> DeliveryResponse:
    """
    Aceita:
      - DeliveryResponse (retorno já pronto do handler)
      - (status, message)
      - (status, message, metadata)
    E normaliza para DeliveryResponse.
    """
    if isinstance(result, DeliveryResponse):
        return result

    if isinstance(result, tuple):
        if len(result) == 2:
            status, message = result
            return _build_response(req, delivery_id, status=status, message=message)
        elif len(result) == 3:
            status, message, metadata = result
            return _build_response(req, delivery_id, status=status, message=message, metadata=metadata)

    # fallback seguro
    return _build_response(req, delivery_id, status="failed", message="Invalid handler return format")


async def deliver_key(req: DeliveryRequest) -> DeliveryResponse:
    """
    Orquestra a entrega.
    - Valida método
    - Invoca handler
    - Normaliza retorno
    - Registra tracking
    """
    delivery_id = str(uuid.uuid4())
    method = (req.delivery_method or "").upper()

    if method not in SUPPORTED_METHODS:
        result = _build_response(
            req,
            delivery_id,
            status="failed",
            message=f"Unsupported delivery method: {req.delivery_method}",
        )
        delivery_tracker[delivery_id] = result
        delivery_attempts[delivery_id] = 1
        return result

    handler = DELIVERY_METHODS.get(method)
    if handler is None:
        result = _build_response(
            req,
            delivery_id,
            status="failed",
            message=f"No handler available for method: {method}",
        )
        delivery_tracker[delivery_id] = result
        delivery_attempts[delivery_id] = 1
        return result

    try:
        # Alguns handlers podem ter assinatura assíncrona, outros síncrona
        handler_result = await handler(req, delivery_id)  # type: ignore
        result = _normalize_handler_result(req, delivery_id, handler_result)
    except Exception as e:
        result = _build_response(
            req,
            delivery_id,
            status="failed",
            message=f"Delivery exception: {e}",
        )

    # Track
    delivery_tracker[delivery_id] = result
    delivery_attempts[delivery_id] = delivery_attempts.get(delivery_id, 0) + 1

    return result


def get_delivery_status(delivery_id: str) -> Optional[DeliveryResponse]:
    """Retorna o status de uma entrega específica."""
    return delivery_tracker.get(delivery_id)


def list_deliveries() -> Dict[str, DeliveryResponse]:
    """Lista todas as entregas trackeadas."""
    return delivery_tracker.copy()