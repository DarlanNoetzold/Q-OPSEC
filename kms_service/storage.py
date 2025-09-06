import json
from datetime import datetime
from cache import redis_client

async def save_session(session_id: str, algorithm: str, key_material: str, expires_at: datetime):
    ttl = int((expires_at - datetime.utcnow()).total_seconds())
    if ttl <= 0:
        ttl = 1  # evita TTL zero/negativo em edge cases

    payload = {
        "algorithm": algorithm,
        "key_material": key_material,
        "expires_at": expires_at.isoformat()
    }
    await redis_client.setex(session_id, ttl, json.dumps(payload))
    return {
        "session_id": session_id,
        "algorithm": algorithm,
        "key_material": key_material,
        "expires_at": expires_at
    }

async def get_session(session_id: str):
    raw = await redis_client.get(session_id)
    if not raw:
        return None
    obj = json.loads(raw)
    return {
        "session_id": session_id,
        "algorithm": obj["algorithm"],
        "key_material": obj["key_material"],
        "expires_at": datetime.fromisoformat(obj["expires_at"])
    }