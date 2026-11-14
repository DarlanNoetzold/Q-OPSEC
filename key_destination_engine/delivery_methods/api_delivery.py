import httpx
from typing import Dict, Any, Tuple
from common.http import SharedAsyncClient, rewrite_host
from config import API_TIMEOUT, DEFAULT_API_DESTINATION
from models import DeliveryRequest

SUPPORTED_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE", "GET"}
_HTTP_CLIENT = SharedAsyncClient(timeout=API_TIMEOUT)


def get_http_client() -> httpx.AsyncClient:
    return _HTTP_CLIENT.get()


async def close_http_client() -> None:
    await _HTTP_CLIENT.aclose()


def _extract_from_metadata(req: DeliveryRequest) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
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
        body = meta["body"]

    return method, headers, body

async def deliver_via_api(req: DeliveryRequest, delivery_id: str):
    url = rewrite_host(str(req.destination or DEFAULT_API_DESTINATION))
    if not url.lower().startswith(("http://", "https://")):
        url = f"http://{url}"

    method, headers, body = _extract_from_metadata(req)
    if method not in SUPPORTED_HTTP_METHODS:
        return (
            "failed",
            f"Unsupported HTTP method: {method}",
            {"delivery_id": delivery_id, "method": method},
        )

    try:
        if method == "GET":
            resp = await get_http_client().get(url, params=body, headers=headers)
        else:
            resp = await get_http_client().request(method, url, json=body, headers=headers)

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