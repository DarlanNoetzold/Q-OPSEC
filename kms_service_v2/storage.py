import os
import time
import threading
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("qopsec.storage")

SESSION_KEY_PREFIX = "qopsec:kms:session:"


class KeySessionStore:
    """Process-local session store. Single-replica only; kept as a fallback."""

    backend = "memory"

    def __init__(self, cleanup_interval_seconds: int = 60):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._request_index: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval_seconds
        self._last_cleanup = time.time()

    def _run_cleanup_if_needed(self):
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time
        expired_session_ids = [
            sid for sid, data in self._sessions.items()
            if data.get("expires_at", 0) < current_time
        ]

        for sid in expired_session_ids:
            session_data = self._sessions.pop(sid, None)
            if session_data:
                rid = session_data.get("request_id")
                if rid and rid in self._request_index:
                    del self._request_index[rid]

        if expired_session_ids:
            logger.info("Cleaned up %d expired sessions", len(expired_session_ids))

    def save(self, session_id: str, request_id: str, algorithm: str,
             key_material: str, expires_at: int, source: str = "unknown") -> bool:
        with self._lock:
            # Keep both indexes consistent when identifiers are reused.
            previous_session = self._sessions.get(session_id)
            if previous_session:
                previous_request_id = previous_session.get("request_id")
                if previous_request_id:
                    self._request_index.pop(previous_request_id, None)

            previous_session_id = self._request_index.get(request_id)
            if previous_session_id and previous_session_id != session_id:
                self._sessions.pop(previous_session_id, None)

            self._sessions[session_id] = {
                "session_id": session_id,
                "request_id": request_id,
                "algorithm": algorithm,
                "key_material": key_material,
                "expires_at": expires_at,
                "source": source,
                "created_at": int(time.time()),
            }
            self._request_index[request_id] = session_id
            self._run_cleanup_if_needed()
        return True

    def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._run_cleanup_if_needed()
            session = self._sessions.get(session_id)
            if not session:
                return None
            if session["expires_at"] < time.time():
                self._sessions.pop(session_id, None)
                rid = session.get("request_id")
                if rid:
                    self._request_index.pop(rid, None)
                return None
            return dict(session)

    def get_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._run_cleanup_if_needed()
            session_id = self._request_index.get(request_id)
            if session_id:
                session = self._sessions.get(session_id)
                if session and session["expires_at"] >= time.time():
                    return dict(session)
                if session:
                    self._sessions.pop(session_id, None)
                    self._request_index.pop(request_id, None)
                return None

            for sid, data in self._sessions.items():
                if data.get("request_id") == request_id and data["expires_at"] >= time.time():
                    self._request_index[request_id] = sid
                    return dict(data)

            return None

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                rid = session.get("request_id")
                if rid:
                    self._request_index.pop(rid, None)
                return True
            return False

    def list_active(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            self._run_cleanup_if_needed()
            current_time = time.time()
            active = []
            for data in self._sessions.values():
                if data["expires_at"] >= current_time:
                    active.append({
                        "session_id": data["session_id"],
                        "request_id": data["request_id"],
                        "algorithm": data["algorithm"],
                        "expires_at": data["expires_at"],
                        "source": data.get("source", "unknown"),
                    })
                if len(active) >= limit:
                    break
            return active

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            self._run_cleanup_if_needed()
            current_time = time.time()
            active_count = sum(
                1 for d in self._sessions.values()
                if d["expires_at"] >= current_time
            )
            return {
                "backend": "memory",
                "total_sessions": len(self._sessions),
                "active_sessions": active_count,
                "request_index_size": len(self._request_index),
            }

    def healthy(self) -> bool:
        return True


class RedisKeySessionStore:
    """Redis-backed session store shared by every KMS replica.

    Sessions are hashes with a TTL equal to their remaining lifetime, so expiry is
    enforced by Redis instead of a per-process sweep. A sorted set indexes sessions
    by ``expires_at`` to keep ``stats``/``list_active`` O(log N) instead of scanning
    the whole keyspace. Key material never leaves the hash and is never logged.
    """

    backend = "redis"

    def __init__(self, client: Any, prefix: str = SESSION_KEY_PREFIX):
        self._client = client
        self._prefix = prefix
        self._index_key = f"{prefix}index"

    def _session_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _request_key(self, request_id: str) -> str:
        return f"{self._prefix}req:{request_id}"

    def _ttl(self, expires_at: int) -> int:
        return max(1, int(expires_at - time.time()))

    def _drop(self, session_id: str, request_id: Optional[str]) -> None:
        pipe = self._client.pipeline()
        pipe.delete(self._session_key(session_id))
        pipe.zrem(self._index_key, session_id)
        if request_id:
            pipe.delete(self._request_key(request_id))
        pipe.execute()

    def save(self, session_id: str, request_id: str, algorithm: str,
             key_material: str, expires_at: int, source: str = "unknown") -> bool:
        previous = self._client.hgetall(self._session_key(session_id))
        previous_request_id = previous.get("request_id") if previous else None
        if previous_request_id and previous_request_id != request_id:
            self._client.delete(self._request_key(previous_request_id))

        previous_session_id = self._client.get(self._request_key(request_id))
        if previous_session_id and previous_session_id != session_id:
            self._drop(previous_session_id, request_id)

        ttl = self._ttl(expires_at)
        pipe = self._client.pipeline()
        pipe.hset(self._session_key(session_id), mapping={
            "session_id": session_id,
            "request_id": request_id,
            "algorithm": algorithm,
            "key_material": key_material,
            "expires_at": int(expires_at),
            "source": source,
            "created_at": int(time.time()),
        })
        pipe.expire(self._session_key(session_id), ttl)
        pipe.set(self._request_key(request_id), session_id, ex=ttl)
        pipe.zadd(self._index_key, {session_id: int(expires_at)})
        pipe.execute()
        return True

    def _load(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self._client.hgetall(self._session_key(session_id))
        if not data:
            return None
        session = {
            "session_id": data.get("session_id", session_id),
            "request_id": data.get("request_id", ""),
            "algorithm": data.get("algorithm", ""),
            "key_material": data.get("key_material", ""),
            "expires_at": int(data.get("expires_at", 0)),
            "source": data.get("source", "unknown"),
            "created_at": int(data.get("created_at", 0)),
        }
        if session["expires_at"] < time.time():
            self._drop(session_id, session.get("request_id"))
            return None
        return session

    def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._load(session_id)

    def get_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        session_id = self._client.get(self._request_key(request_id))
        if not session_id:
            return None
        return self._load(session_id)

    def delete(self, session_id: str) -> bool:
        session = self._client.hgetall(self._session_key(session_id))
        if not session:
            return False
        self._drop(session_id, session.get("request_id"))
        return True

    def list_active(self, limit: int = 100) -> List[Dict[str, Any]]:
        now = int(time.time())
        self._client.zremrangebyscore(self._index_key, "-inf", now)
        session_ids = self._client.zrangebyscore(
            self._index_key, now + 1, "+inf", start=0, num=max(1, limit)
        )
        active = []
        for session_id in session_ids:
            session = self._load(session_id)
            if session:
                active.append({
                    "session_id": session["session_id"],
                    "request_id": session["request_id"],
                    "algorithm": session["algorithm"],
                    "expires_at": session["expires_at"],
                    "source": session.get("source", "unknown"),
                })
        return active

    def stats(self) -> Dict[str, Any]:
        now = int(time.time())
        self._client.zremrangebyscore(self._index_key, "-inf", now)
        return {
            "backend": "redis",
            "total_sessions": int(self._client.zcard(self._index_key)),
            "active_sessions": int(self._client.zcard(self._index_key)),
            "request_index_size": int(self._client.zcard(self._index_key)),
        }

    def healthy(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


def build_session_store() -> Any:
    """Select the session backend from the environment.

    ``QOPSEC_KMS_SESSION_BACKEND`` accepts ``redis``, ``memory`` or ``auto``
    (default). ``auto`` uses Redis when reachable and degrades to the in-memory
    store so a single-replica deployment still boots. ``redis`` fails fast.
    """
    backend = os.environ.get("QOPSEC_KMS_SESSION_BACKEND", "auto").strip().lower()
    if backend == "memory":
        return KeySessionStore()

    url = os.environ.get("QOPSEC_REDIS_URL")
    if not url:
        if backend == "redis":
            raise RuntimeError("QOPSEC_REDIS_URL is required when QOPSEC_KMS_SESSION_BACKEND=redis")
        logger.warning("QOPSEC_REDIS_URL not set; KMS sessions stay in process memory")
        return KeySessionStore()

    try:
        import redis

        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        client.ping()
        store = RedisKeySessionStore(
            client,
            prefix=os.environ.get("QOPSEC_KMS_SESSION_PREFIX", SESSION_KEY_PREFIX),
        )
        logger.info("KMS session store using Redis backend")
        return store
    except Exception as exc:
        if backend == "redis":
            raise
        logger.warning(
            "Redis session backend unavailable (%s); falling back to in-memory store",
            type(exc).__name__,
        )
        return KeySessionStore()


session_store = build_session_store()
