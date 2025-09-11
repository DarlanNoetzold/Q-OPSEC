"""
Key Session Storage for KMS.

Supports multiple backends:
- Redis (production - recommended)
- SQLite (development/testing)
- In-memory (temporary fallback)

This module:
- Always stores expires_at internally as epoch seconds (int)
- Converts datetime -> epoch on save
- Converts epoch -> datetime on fetch (for API models expecting datetime)
"""

import json
import time
import sqlite3
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Backend configuration
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "memory")  # "redis", "sqlite", "memory"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SQLITE_PATH = os.getenv("SQLITE_PATH", "kms_sessions.db")

# Internal state
_memory_store: Dict[str, Dict[str, Any]] = {}
_redis_client = None
_sqlite_conn: Optional[sqlite3.Connection] = None


def _to_epoch(value) -> int:
    """Convert datetime or int to epoch seconds (UTC)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # Assume UTC if naive
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())
    return int(value)


def _to_datetime(epoch: int) -> datetime:
    """Convert epoch seconds to UTC datetime."""
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc)


def _init_redis() -> bool:
    """Initialize Redis connection (lazy)."""
    global _redis_client
    if _redis_client is not None:
        return True
    try:
        import redis  # type: ignore
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        print(f"[Storage] Redis connected: {REDIS_URL}")
        return True
    except Exception as e:
        print(f"[Storage] Redis connection failed: {e}")
        _redis_client = None
        return False


def _init_sqlite() -> bool:
    """Initialize SQLite database (lazy)."""
    global _sqlite_conn
    if _sqlite_conn is not None:
        return True
    try:
        _sqlite_conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        _sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS key_sessions (
                session_id TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                key_material TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                source TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        _sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON key_sessions(expires_at)
        """)
        _sqlite_conn.commit()
        print(f"[Storage] SQLite initialized: {SQLITE_PATH}")
        return True
    except Exception as e:
        print(f"[Storage] SQLite init error: {e}")
        _sqlite_conn = None
        return False


def _cleanup_expired():
    """Remove expired sessions periodically."""
    current_time = int(time.time())

    if STORAGE_BACKEND == "redis" and _redis_client:
        # Redis uses TTL; no manual cleanup required
        return

    if STORAGE_BACKEND == "sqlite" and _sqlite_conn:
        try:
            cursor = _sqlite_conn.execute(
                "DELETE FROM key_sessions WHERE expires_at < ?",
                (current_time,)
            )
            deleted = cursor.rowcount
            _sqlite_conn.commit()
            if deleted and deleted > 0:
                print(f"[Storage] Removed {deleted} expired sessions from SQLite")
        except Exception as e:
            print(f"[Storage] SQLite cleanup error: {e}")
        return

    if STORAGE_BACKEND == "memory":
        expired_keys = [
            k for k, v in _memory_store.items()
            if int(v.get("expires_at", 0)) < current_time
        ]
        for key in expired_keys:
            _memory_store.pop(key, None)
        if expired_keys:
            print(f"[Storage] Removed {len(expired_keys)} expired sessions from memory")


async def save_session(session_data: Dict[str, Any]) -> bool:
    """
    Save a key session.

    Args:
        session_data: Dict with keys:
            - session_id (str)
            - algorithm (str)
            - key_material (str)
            - expires_at (datetime or int epoch)
            - source (str)

    Returns:
        True if saved successfully.
    """
    session_id = session_data["session_id"]

    # Normalize expires_at -> epoch
    expires_epoch = _to_epoch(session_data["expires_at"])

    # Prepare a serializable dict for storage
    data_to_store = dict(session_data)
    data_to_store["expires_at"] = expires_epoch

    ttl_seconds = max(1, expires_epoch - int(time.time()))

    try:
        if STORAGE_BACKEND == "redis":
            if not _init_redis():
                return await _save_memory(data_to_store)

            # Store with TTL
            _redis_client.setex(
                f"kms:session:{session_id}",
                ttl_seconds,
                json.dumps(data_to_store)
            )
            return True

        if STORAGE_BACKEND == "sqlite":
            if not _init_sqlite():
                return await _save_memory(data_to_store)

            _sqlite_conn.execute("""
                INSERT OR REPLACE INTO key_sessions 
                (session_id, algorithm, key_material, expires_at, source)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                data_to_store["algorithm"],
                data_to_store["key_material"],
                expires_epoch,
                data_to_store.get("source", "unknown")
            ))
            _sqlite_conn.commit()

            # Periodic cleanup
            if hash(session_id) % 100 == 0:
                _cleanup_expired()

            return True

        # Default: memory
        return await _save_memory(data_to_store)

    except Exception as e:
        print(f"[Storage] Error saving session {session_id}: {e}")
        # Fallback to memory on error
        return await _save_memory(data_to_store)


async def _save_memory(session_data: Dict[str, Any]) -> bool:
    """Save session data in memory (fallback)."""
    session_id = session_data["session_id"]
    _memory_store[session_id] = session_data

    # Periodic cleanup
    if len(_memory_store) % 50 == 0:
        _cleanup_expired()

    return True


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a key session by id.

    Returns:
        Dict with:
            - session_id (str)
            - algorithm (str)
            - key_material (str)
            - expires_at (datetime)  [converted from epoch]
            - source (str)
        or None if not found/expired
    """
    current_time = int(time.time())

    try:
        if STORAGE_BACKEND == "redis" and _redis_client:
            data = _redis_client.get(f"kms:session:{session_id}")
            if not data:
                return None
            session_data = json.loads(data)
            if int(session_data.get("expires_at", 0)) > current_time:
                # Convert epoch -> datetime (UTC)
                session_data["expires_at"] = _to_datetime(session_data["expires_at"])
                return session_data
            return None

        if STORAGE_BACKEND == "sqlite" and _sqlite_conn:
            cursor = _sqlite_conn.execute("""
                SELECT session_id, algorithm, key_material, expires_at, source
                FROM key_sessions 
                WHERE session_id = ? AND expires_at > ?
            """, (session_id, current_time))
            row = cursor.fetchone()
            if row:
                return {
                    "session_id": row[0],
                    "algorithm": row[1],
                    "key_material": row[2],
                    "expires_at": _to_datetime(row[3]),
                    "source": row[4] or "unknown"
                }
            return None

        # memory
        session_data = _memory_store.get(session_id)
        if session_data and int(session_data.get("expires_at", 0)) > current_time:
            session_copy = dict(session_data)
            session_copy["expires_at"] = _to_datetime(session_copy["expires_at"])
            return session_copy
        elif session_data:
            # Remove if expired
            _memory_store.pop(session_id, None)
            return None

        return None

    except Exception as e:
        print(f"[Storage] Error retrieving session {session_id}: {e}")
        return None


async def delete_session(session_id: str) -> bool:
    """
    Delete (revoke) a key session.

    Returns:
        True if the session was deleted.
    """
    try:
        if STORAGE_BACKEND == "redis" and _redis_client:
            result = _redis_client.delete(f"kms:session:{session_id}")
            return result > 0

        if STORAGE_BACKEND == "sqlite" and _sqlite_conn:
            cursor = _sqlite_conn.execute(
                "DELETE FROM key_sessions WHERE session_id = ?",
                (session_id,)
            )
            _sqlite_conn.commit()
            return cursor.rowcount > 0

        # memory
        if session_id in _memory_store:
            _memory_store.pop(session_id, None)
            return True
        return False

    except Exception as e:
        print(f"[Storage] Error deleting session {session_id}: {e}")
        return False


async def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    """
    List active sessions (for debug/admin).
    """
    current_time = int(time.time())
    try:
        if STORAGE_BACKEND == "redis" and _redis_client:
            keys = _redis_client.keys("kms:session:*")
            sessions = []
            for key in keys[:limit]:
                data = _redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    if int(session_data.get("expires_at", 0)) > current_time:
                        sessions.append({
                            "session_id": session_data["session_id"],
                            "algorithm": session_data["algorithm"],
                            "expires_at": session_data["expires_at"],
                            "source": session_data.get("source", "unknown")
                        })
            return sessions

        if STORAGE_BACKEND == "sqlite" and _sqlite_conn:
            cursor = _sqlite_conn.execute("""
                SELECT session_id, algorithm, expires_at, source
                FROM key_sessions 
                WHERE expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (current_time, limit))
            return [
                {
                    "session_id": row[0],
                    "algorithm": row[1],
                    "expires_at": row[2],
                    "source": row[3] or "unknown"
                }
                for row in cursor.fetchall()
            ]

        # memory
        sessions = []
        for session_data in list(_memory_store.values())[:limit]:
            if int(session_data.get("expires_at", 0)) > current_time:
                sessions.append({
                    "session_id": session_data["session_id"],
                    "algorithm": session_data["algorithm"],
                    "expires_at": session_data["expires_at"],
                    "source": session_data.get("source", "unknown")
                })
        return sessions

    except Exception as e:
        print(f"[Storage] Error listing sessions: {e}")
        return []


def get_storage_stats() -> Dict[str, Any]:
    """
    Return storage statistics for the current backend.
    """
    current_time = int(time.time())
    try:
        if STORAGE_BACKEND == "redis" and _redis_client:
            keys = _redis_client.keys("kms:session:*")
            return {
                "backend": "redis",
                "total_sessions": len(keys),
                "redis_info": _redis_client.info("memory")
            }

        if STORAGE_BACKEND == "sqlite" and _sqlite_conn:
            cursor = _sqlite_conn.execute(
                "SELECT COUNT(*) FROM key_sessions WHERE expires_at > ?",
                (current_time,)
            )
            active_count = cursor.fetchone()[0]
            cursor = _sqlite_conn.execute("SELECT COUNT(*) FROM key_sessions")
            total_count = cursor.fetchone()[0]
            return {
                "backend": "sqlite",
                "active_sessions": active_count,
                "total_sessions": total_count,
                "database_path": SQLITE_PATH
            }

        # memory
        active_sessions = sum(
            1 for v in _memory_store.values()
            if int(v.get("expires_at", 0)) > current_time
        )
        return {
            "backend": "memory",
            "active_sessions": active_sessions,
            "total_sessions": len(_memory_store)
        }

    except Exception as e:
        return {
            "backend": STORAGE_BACKEND,
            "error": str(e)
        }


# Auto-initialize based on backend
if STORAGE_BACKEND == "redis":
    _init_redis()
elif STORAGE_BACKEND == "sqlite":
    _init_sqlite()

print(f"[Storage] Backend configured: {STORAGE_BACKEND}")