"""
Sistema de armazenamento de chaves para o KMS.

Suporta múltiplos backends:
- Redis (produção - recomendado)
- SQLite (desenvolvimento/teste)
- In-memory (fallback temporário)
"""

import json
import time
import sqlite3
import os
from typing import Optional, Dict, Any
from datetime import datetime

# Configuração do backend de storage
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "memory")  # "redis", "sqlite", "memory"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SQLITE_PATH = os.getenv("SQLITE_PATH", "kms_sessions.db")

# Storage backends
_memory_store = {}
_redis_client = None
_sqlite_conn = None


def _init_redis():
    """Inicializa conexão Redis"""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            # Testa conexão
            _redis_client.ping()
            print(f"[Storage] Redis conectado: {REDIS_URL}")
            return True
        except Exception as e:
            print(f"[Storage] Erro ao conectar Redis: {e}")
            _redis_client = None
            return False
    return True


def _init_sqlite():
    """Inicializa banco SQLite"""
    global _sqlite_conn
    if _sqlite_conn is None:
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
            print(f"[Storage] SQLite inicializado: {SQLITE_PATH}")
            return True
        except Exception as e:
            print(f"[Storage] Erro ao inicializar SQLite: {e}")
            _sqlite_conn = None
            return False
    return True


def _cleanup_expired():
    """Remove sessões expiradas (executado automaticamente)"""
    current_time = int(time.time())

    if STORAGE_BACKEND == "redis" and _redis_client:
        # Redis tem TTL automático, não precisa cleanup manual
        pass
    elif STORAGE_BACKEND == "sqlite" and _sqlite_conn:
        try:
            cursor = _sqlite_conn.execute(
                "DELETE FROM key_sessions WHERE expires_at < ?",
                (current_time,)
            )
            deleted = cursor.rowcount
            _sqlite_conn.commit()
            if deleted > 0:
                print(f"[Storage] Removidas {deleted} sessões expiradas do SQLite")
        except Exception as e:
            print(f"[Storage] Erro no cleanup SQLite: {e}")
    elif STORAGE_BACKEND == "memory":
        expired_keys = [
            k for k, v in _memory_store.items()
            if v.get("expires_at", 0) < current_time
        ]
        for key in expired_keys:
            del _memory_store[key]
        if expired_keys:
            print(f"[Storage] Removidas {len(expired_keys)} sessões expiradas da memória")


async def save_session(session_data: Dict[str, Any]) -> bool:
    """
    Salva uma sessão de chave.

    Args:
        session_data: Dict com session_id, algorithm, key_material, expires_at, source

    Returns:
        bool: True se salvou com sucesso
    """
    session_id = session_data["session_id"]
    expires_at = session_data["expires_at"]
    ttl_seconds = max(1, expires_at - int(time.time()))

    try:
        if STORAGE_BACKEND == "redis":
            if not _init_redis():
                # Fallback para memory se Redis falhar
                return await _save_memory(session_data)

            # Salva no Redis com TTL automático
            _redis_client.setex(
                f"kms:session:{session_id}",
                ttl_seconds,
                json.dumps(session_data)
            )
            return True

        elif STORAGE_BACKEND == "sqlite":
            if not _init_sqlite():
                return await _save_memory(session_data)

            _sqlite_conn.execute("""
                INSERT OR REPLACE INTO key_sessions 
                (session_id, algorithm, key_material, expires_at, source)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                session_data["algorithm"],
                session_data["key_material"],
                expires_at,
                session_data.get("source", "unknown")
            ))
            _sqlite_conn.commit()

            # Cleanup periódico (a cada 100 inserções)
            if hash(session_id) % 100 == 0:
                _cleanup_expired()

            return True

        else:  # memory
            return await _save_memory(session_data)

    except Exception as e:
        print(f"[Storage] Erro ao salvar sessão {session_id}: {e}")
        # Fallback para memory em caso de erro
        return await _save_memory(session_data)


async def _save_memory(session_data: Dict[str, Any]) -> bool:
    """Salva na memória (fallback)"""
    session_id = session_data["session_id"]
    _memory_store[session_id] = session_data

    # Cleanup periódico
    if len(_memory_store) % 50 == 0:
        _cleanup_expired()

    return True


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Recupera uma sessão de chave.

    Args:
        session_id: ID da sessão

    Returns:
        Dict com dados da sessão ou None se não encontrada/expirada
    """
    current_time = int(time.time())

    try:
        if STORAGE_BACKEND == "redis" and _redis_client:
            data = _redis_client.get(f"kms:session:{session_id}")
            if data:
                session_data = json.loads(data)
                # Verifica expiração (redundante, mas seguro)
                if session_data.get("expires_at", 0) > current_time:
                    return session_data
            return None

        elif STORAGE_BACKEND == "sqlite" and _sqlite_conn:
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
                    "expires_at": row[3],
                    "source": row[4] or "unknown"
                }
            return None

        else:  # memory
            session_data = _memory_store.get(session_id)
            if session_data and session_data.get("expires_at", 0) > current_time:
                return session_data
            elif session_data:
                # Remove se expirada
                del _memory_store[session_id]
            return None

    except Exception as e:
        print(f"[Storage] Erro ao recuperar sessão {session_id}: {e}")
        return None


async def delete_session(session_id: str) -> bool:
    """
    Remove uma sessão de chave (revogação).

    Args:
        session_id: ID da sessão

    Returns:
        bool: True se removeu com sucesso
    """
    try:
        if STORAGE_BACKEND == "redis" and _redis_client:
            result = _redis_client.delete(f"kms:session:{session_id}")
            return result > 0

        elif STORAGE_BACKEND == "sqlite" and _sqlite_conn:
            cursor = _sqlite_conn.execute(
                "DELETE FROM key_sessions WHERE session_id = ?",
                (session_id,)
            )
            _sqlite_conn.commit()
            return cursor.rowcount > 0

        else:  # memory
            if session_id in _memory_store:
                del _memory_store[session_id]
                return True
            return False

    except Exception as e:
        print(f"[Storage] Erro ao deletar sessão {session_id}: {e}")
        return False


async def list_sessions(limit: int = 100) -> list:
    """
    Lista sessões ativas (para debug/admin).

    Args:
        limit: Máximo de sessões a retornar

    Returns:
        Lista de sessões ativas
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
                    if session_data.get("expires_at", 0) > current_time:
                        sessions.append({
                            "session_id": session_data["session_id"],
                            "algorithm": session_data["algorithm"],
                            "expires_at": session_data["expires_at"],
                            "source": session_data.get("source", "unknown")
                        })
            return sessions

        elif STORAGE_BACKEND == "sqlite" and _sqlite_conn:
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

        else:  # memory
            sessions = []
            for session_data in list(_memory_store.values())[:limit]:
                if session_data.get("expires_at", 0) > current_time:
                    sessions.append({
                        "session_id": session_data["session_id"],
                        "algorithm": session_data["algorithm"],
                        "expires_at": session_data["expires_at"],
                        "source": session_data.get("source", "unknown")
                    })
            return sessions

    except Exception as e:
        print(f"[Storage] Erro ao listar sessões: {e}")
        return []


def get_storage_stats() -> Dict[str, Any]:
    """
    Retorna estatísticas do storage.

    Returns:
        Dict com estatísticas do backend atual
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

        elif STORAGE_BACKEND == "sqlite" and _sqlite_conn:
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

        else:  # memory
            active_sessions = sum(
                1 for v in _memory_store.values()
                if v.get("expires_at", 0) > current_time
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


# Inicialização automática baseada na configuração
if STORAGE_BACKEND == "redis":
    _init_redis()
elif STORAGE_BACKEND == "sqlite":
    _init_sqlite()

print(f"[Storage] Backend configurado: {STORAGE_BACKEND}")