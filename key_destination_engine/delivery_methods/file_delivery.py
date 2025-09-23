# delivery_methods/file_delivery.py
import os
from typing import Tuple, Dict, Any
from config import FILE_DELIVERY_BASE_PATH
from models import DeliveryRequest

async def deliver_via_file(req: DeliveryRequest, delivery_id: str):
    dest = req.destination
    if not dest or dest.strip() == "":
        dest = os.path.join(FILE_DELIVERY_BASE_PATH, f"{req.session_id}.key")

    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(req.key_material)
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