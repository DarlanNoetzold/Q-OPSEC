import os
import json
from datetime import datetime
from models import DeliveryRequest, DeliveryResponse
from config import FILE_DELIVERY_BASE_PATH


async def deliver_via_file(req: DeliveryRequest, delivery_id: str) -> DeliveryResponse:
    """
    Deliver key material by writing to a secure file.
    """
    try:
        # Ensure base directory exists
        os.makedirs(FILE_DELIVERY_BASE_PATH, exist_ok=True)

        # Create filename
        filename = f"{req.session_id}_{delivery_id}.json"
        filepath = os.path.join(FILE_DELIVERY_BASE_PATH, filename)

        # Prepare file content
        content = {
            "session_id": req.session_id,
            "algorithm": req.algorithm,
            "key_material": req.key_material,
            "expires_at": req.expires_at.isoformat(),
            "delivery_id": delivery_id,
            "destination": req.destination,
            "delivered_at": datetime.utcnow().isoformat()
        }

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=2)

        # Set secure permissions (owner read/write only)
        os.chmod(filepath, 0o600)

        print(f"[KDE] Key written to file: {filepath}")

        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="delivered",
            delivery_method="FILE",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"Key securely written to {filepath}",
            metadata={"filepath": filepath, "permissions": "0600"}
        )

    except Exception as e:
        return DeliveryResponse(
            session_id=req.session_id,
            destination=req.destination,
            status="failed",
            delivery_method="FILE",
            timestamp=datetime.utcnow(),
            delivery_id=delivery_id,
            message=f"File delivery failed: {str(e)}"
        )