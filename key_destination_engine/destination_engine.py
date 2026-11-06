import asyncio
import os
import uuid
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union

from common.ttl_cache import TTLCache

from models import DeliveryRequest, DeliveryResponse
from config import SUPPORTED_METHODS

try:
    from delivery_methods.api_delivery import deliver_via_api
except Exception:
    deliver_via_api = None

try:
    from delivery_methods.mqtt_delivery import deliver_via_mqtt
except Exception:
    deliver_via_mqtt = None

try:
    from delivery_methods.hsm_delivery import deliver_via_hsm
except Exception:
    deliver_via_hsm = None

try:
    from delivery_methods.file_delivery import deliver_via_file
except Exception:
    deliver_via_file = None

DELIVERY_METHODS: Dict[str, Optional[object]] = {
    "API": deliver_via_api,
    "MQTT": deliver_via_mqtt,
    "HSM": deliver_via_hsm,
    "FILE": deliver_via_file,
}

DELIVERY_TTL_SECONDS = float(os.environ.get("QOPSEC_DELIVERY_TTL_SECONDS", "3600"))
MAX_TRACKED_DELIVERIES = int(os.environ.get("QOPSEC_DELIVERY_MAX_ENTRIES", "1000"))

delivery_tracker: TTLCache = TTLCache(max_entries=MAX_TRACKED_DELIVERIES)
delivery_attempts: TTLCache = TTLCache(max_entries=MAX_TRACKED_DELIVERIES)
_delivery_ids: deque[str] = deque(maxlen=MAX_TRACKED_DELIVERIES)
_tracker_lock = threading.Lock()
_background_tasks: set[asyncio.Task] = set()


def _record(delivery_id: str, result: DeliveryResponse, attempts: Optional[int] = None) -> None:
    with _tracker_lock:
        delivery_tracker.set(delivery_id, result, DELIVERY_TTL_SECONDS)
        current = delivery_attempts.get(delivery_id) or 0
        delivery_attempts.set(
            delivery_id,
            current + 1 if attempts is None else attempts,
            DELIVERY_TTL_SECONDS,
        )
        _delivery_ids.append(delivery_id)



def _build_response(
    req: DeliveryRequest,
    delivery_id: str,
    status: str,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DeliveryResponse:
    return DeliveryResponse(
        session_id=req.session_id,
        request_id=req.request_id or "req-unknown",
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

    if isinstance(result, DeliveryResponse):
        return result

    if isinstance(result, tuple):
        if len(result) == 2:
            status, message = result
            return _build_response(req, delivery_id, status=status, message=message)
        elif len(result) == 3:
            status, message, metadata = result
            return _build_response(req, delivery_id, status=status, message=message, metadata=metadata)

    return _build_response(req, delivery_id, status="failed", message="Invalid handler return format")


async def deliver_key(req: DeliveryRequest, delivery_id: Optional[str] = None) -> DeliveryResponse:

    delivery_id = delivery_id or str(uuid.uuid4())
    method = (req.delivery_method or "").upper()

    if method not in SUPPORTED_METHODS:
        result = _build_response(
            req,
            delivery_id,
            status="failed",
            message=f"Unsupported delivery method: {req.delivery_method}",
        )
        _record(delivery_id, result, attempts=1)
        return result

    handler = DELIVERY_METHODS.get(method)
    if handler is None:
        result = _build_response(
            req,
            delivery_id,
            status="failed",
            message=f"No handler available for method: {method}",
        )
        _record(delivery_id, result, attempts=1)
        return result

    try:
        handler_result = await handler(req, delivery_id)
        result = _normalize_handler_result(req, delivery_id, handler_result)
    except Exception as e:
        result = _build_response(
            req,
            delivery_id,
            status="failed",
            message=f"Delivery exception: {e}",
        )

    _record(delivery_id, result)

    return result


def queue_delivery(req: DeliveryRequest) -> DeliveryResponse:
    """Accept the delivery and process it in the background, returning the tracking id."""
    delivery_id = str(uuid.uuid4())
    queued = _build_response(req, delivery_id, status="queued", message="Delivery queued")
    _record(delivery_id, queued, attempts=0)
    task = asyncio.create_task(deliver_key(req, delivery_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return queued


def get_delivery_status(delivery_id: str) -> Optional[DeliveryResponse]:
    return delivery_tracker.get(delivery_id)


def list_deliveries() -> Dict[str, DeliveryResponse]:
    with _tracker_lock:
        delivery_ids = list(_delivery_ids)
    deliveries: Dict[str, DeliveryResponse] = {}
    for delivery_id in reversed(delivery_ids):
        if delivery_id in deliveries:
            continue
        result = delivery_tracker.get(delivery_id)
        if result is not None:
            deliveries[delivery_id] = result
    return deliveries