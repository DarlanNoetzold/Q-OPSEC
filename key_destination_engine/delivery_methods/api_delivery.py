# delivery_methods/api_delivery.py
import httpx
from typing import Dict, Any, Tuple, Optional
from config import API_TIMEOUT
from models import DeliveryRequest

def _extract_from_metadata(req: DeliveryRequest) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    # Método HTTP (POST por padrão)
    method = "POST"
    headers: Dict[str, str] = {}
    body: Dict[str, Any] = {
        "session_id": req.session_id,
        "request_id": req.request_id,
        "algorithm": req.algorithm,
        "expires_at": req.expires_at,
        "key_material": req.key_material,
    }

    meta = req.metadata or {}
    if isinstance(meta.get("method"), str):
        method = meta["method"].upper() or "POST"

    if isinstance(meta.get("headers"), dict):
        headers = {str(k): str(v) for k, v in meta["headers"].items()}

    if isinstance(meta.get("body"), dict):
        # Permite sobrescrever o body padrão
        body = meta["body"]

    return method, headers, body

async def deliver_via_api(req: DeliveryRequest, delivery_id: str):
    # Inicializa url no início para evitar 'unbound local variable'
    url = req.destination

    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return (
            "failed",
            f"Invalid API destination URL: {url}",
            {"delivery_id": delivery_id, "reason": "invalid_url"},
        )

    method, headers, body = _extract_from_metadata(req)

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            if method == "POST":
                resp = await client.post(url, json=body, headers=headers)
            elif method == "PUT":
                resp = await client.put(url, json=body, headers=headers)
            elif method == "PATCH":
                resp = await client.patch(url, json=body, headers=headers)
            elif method == "DELETE":
                resp = await client.request("DELETE", url, json=body, headers=headers)
            elif method == "GET":
                # Para GET, enviamos params ao invés de body
                resp = await client.get(url, params=body, headers=headers)
            else:
                return (
                    "failed",
                    f"Unsupported HTTP method: {method}",
                    {"delivery_id": delivery_id, "method": method},
                )

        ok = 200 <= resp.status_code < 300
        if ok:
            return (
                "delivered",
                f"Delivered via API to {url}",
                {
                    "delivery_id": delivery_id,
                    "http_status": resp.status_code,
                    "response_preview": (resp.text[:200] if resp.text else ""),
                },
            )
        else:
            return (
                "failed",
                f"API responded {resp.status_code}",
                {
                    "delivery_id": delivery_id,
                    "http_status": resp.status_code,
                    "response_preview": (resp.text[:200] if resp.text else ""),
                },
            )

    except Exception as e:
        return (
            "failed",
            f"API delivery exception: {e}",
            {"delivery_id": delivery_id, "url": url},
        )