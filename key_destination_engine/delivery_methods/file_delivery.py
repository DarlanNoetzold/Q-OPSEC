import asyncio
import os
from config import FILE_DELIVERY_BASE_PATH
from models import DeliveryRequest


def _resolve_destination(dest: str, session_id: str) -> str:
    base = os.path.realpath(FILE_DELIVERY_BASE_PATH)
    if not dest or dest.strip() == "":
        dest = f"{session_id}.key"
    target = os.path.realpath(os.path.join(base, dest))
    if os.path.commonpath([base, target]) != base:
        raise ValueError(f"Destination must be inside {base}")
    return target


def _write_key(path: str, key_material: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(key_material)


async def deliver_via_file(req: DeliveryRequest, delivery_id: str):
    dest = req.destination
    try:
        dest = _resolve_destination(req.destination, req.session_id)
        await asyncio.to_thread(_write_key, dest, req.key_material)
        return (
            "delivered",
            f"Key written to file: {dest}",
            {"delivery_id": delivery_id, "path": dest},
        )
    except Exception as e:
        return (
            "failed",
            f"File delivery error: {e}",
            {"delivery_id": delivery_id, "path": dest},
        )
