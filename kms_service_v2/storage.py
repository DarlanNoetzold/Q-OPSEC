import time
import threading
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("qopsec.storage")


class KeySessionStore:

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


session_store = KeySessionStore()
