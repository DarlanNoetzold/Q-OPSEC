from redis.asyncio import Redis

redis_client = Redis.from_url(
    "redis://localhost:6379",
    decode_responses=True,
)
async def cache_set(session_id: str, key_material: str, ttl: int):
    await redis_client.setex(session_id, ttl, key_material)

async def cache_get(session_id: str):
    return await redis_client.get(session_id)