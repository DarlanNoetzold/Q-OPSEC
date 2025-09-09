# storage.py
import os, json, asyncio
from datetime import datetime, timedelta

USE_INMEM = os.getenv("USE_INMEMORY_STORAGE", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

_inmem = {}
_redis = None

async def _get_redis():
    global _redis
    if USE_INMEM:
        return None
    if _redis is None:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            # teste de conexão
            await _redis.ping()
        except Exception:
            _redis = None
    return _redis

async def save_session(session_id: str, algorithm: str, key_material: str, expires):
    ttl = int((expires - datetime.utcnow()).total_seconds())
    payload = {"algorithm": algorithm, "key_material": key_material, "expires": expires.isoformat()}
    r = await _get_redis()
    if r:
        await r.setex(session_id, ttl, json.dumps(payload))
    else:
        _inmem[session_id] = (payload, datetime.utcnow() + timedelta(seconds=ttl))

async def get_session(session_id: str):
    r = await _get_redis()
    if r:
        raw = await r.get(session_id)
        return json.loads(raw) if raw else None
    else:
        v = _inmem.get(session_id)
        if not v:
            return None
        payload, exp = v
        return payload if exp > datetime.utcnow() else None