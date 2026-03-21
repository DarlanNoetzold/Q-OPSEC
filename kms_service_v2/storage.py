"""
Key Session Storage for KMS - In-Memory backend with TTL support.

Stores expires_at as epoch seconds (int).
"""

import time
from typing import Optional, Dict, Any, List

# In-memory store
_memory_store: Dict[str, Dict[str, Any]] = {}
# Index: request_id -> session_id
_request_index: Dict[str, str] = {}


def _cleanup_expired():
    """Remove expired sessions."""
    current_time = int(time.time())
    expired_keys = [
        k for k, v in _memory_store.items()
        if int(v.get("expires_at", 0)) < current_time
    ]
    for key in expired_keys:
        sess = _memory_store.pop(key, None)
        if sess:
            rid = sess.get("request_id", "")
            _request_index.pop(rid, None)
    if expired_keys:
        print(f"[Storage] Removed {len(expired_keys)} expired sessions")


async def save_session(
    session_id: str,
    request_id: str,
    algorithm: str,
    key_material: str,
    expires_at: int,
    source: str = "unknown"
) -> bool:
    """Save a key session to memory."""
    session_data = {
        "session_id": session_id,
        "request_id": request_id,
        "algorithm": algorithm,
        "key_material": key_material,
        "expires_at": expires_at,
        "source": source,
    }
    _memory_store[session_id] = session_data
    if request_id:
        _request_index[request_id] = session_id

    # Periodic cleanup
    if len(_memory_store) % 50 == 0:
        _cleanup_expired()

    return True


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a key session by session_id OR request_id.
    Returns dict or None if not found/expired.
    """
    current_time = int(time.time())

    # Try direct session_id lookup
    session_data = _memory_store.get(session_id)
    if session_data and int(session_data.get("expires_at", 0)) > current_time:
        return dict(session_data)
    elif session_data:
        _memory_store.pop(session_id, None)

    # Try request_id index
    mapped_sid = _request_index.get(session_id)
    if mapped_sid:
        session_data = _memory_store.get(mapped_sid)
        if session_data and int(session_data.get("expires_at", 0)) > current_time:
            return dict(session_data)

    # Fallback: linear search by request_id
    for sess in _memory_store.values():
        if sess.get("request_id") == session_id and int(sess.get("expires_at", 0)) > current_time:
            return dict(sess)

    return None


async def get_session_by_request(request_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a key session by request_id."""
    current_time = int(time.time())

    # Check index first
    mapped_sid = _request_index.get(request_id)
    if mapped_sid:
        session_data = _memory_store.get(mapped_sid)
        if session_data and int(session_data.get("expires_at", 0)) > current_time:
            return dict(session_data)

    # Linear search fallback
    for session_data in _memory_store.values():
        if (session_data.get("request_id") == request_id and
                int(session_data.get("expires_at", 0)) > current_time):
            return dict(session_data)

    return None


async def delete_session(session_id: str) -> bool:
    """Delete a key session."""
    if session_id in _memory_store:
        sess = _memory_store.pop(session_id)
        rid = sess.get("request_id", "")
        _request_index.pop(rid, None)
        return True
    return False


async def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    """List active sessions."""
    current_time = int(time.time())
    sessions = []
    for session_data in list(_memory_store.values())[:limit]:
        if int(session_data.get("expires_at", 0)) > current_time:
            sessions.append({
                "session_id": session_data["session_id"],
                "algorithm": session_data["algorithm"],
                "expires_at": session_data["expires_at"],
                "source": session_data.get("source", "unknown"),
            })
    return sessions


print("[Storage] Backend configured: memory")
