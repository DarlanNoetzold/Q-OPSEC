import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = os.environ.get(
    "QOPSEC_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/kms_db",
)
DATABASE_READ_URL = os.environ.get("QOPSEC_DATABASE_READ_URL", DATABASE_URL)

POOL_SIZE = int(os.environ.get("QOPSEC_DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.environ.get("QOPSEC_DB_MAX_OVERFLOW", "20"))
POOL_RECYCLE = int(os.environ.get("QOPSEC_DB_POOL_RECYCLE_SECONDS", "1800"))

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=POOL_RECYCLE,
)

read_engine = create_engine(
    DATABASE_READ_URL,
    future=True,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=POOL_RECYCLE,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

ReadSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=read_engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_read_db():
    db = ReadSessionLocal()
    try:
        yield db
    finally:
        db.close()


def pool_status():
    return {
        "write": {
            "pool_size": engine.pool.size(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
        },
        "read": {
            "pool_size": read_engine.pool.size(),
            "checked_out": read_engine.pool.checkedout(),
            "overflow": read_engine.pool.overflow(),
        },
        "replica_configured": DATABASE_READ_URL != DATABASE_URL,
    }

def init_db(create_if_missing: bool = False):
    """
    Inicializa o schema.
    - create_if_missing=True: tenta criar tabelas que não existam (não faz migração).
    - Se a tabela já existe, é recomendado usar migração (Alembic) para novas colunas.
    """
    if create_if_missing:
        Base.metadata.create_all(bind=engine)

def migrate_add_request_id():
    """
    Migração simples e idempotente para adicionar coluna e índice de request_id,
    caso você já tenha a tabela criada sem essa coluna.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='key_sessions' AND column_name='request_id'
                ) THEN
                    ALTER TABLE key_sessions ADD COLUMN request_id TEXT;
                END IF;
            END$$;
        """))
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 
                    FROM pg_indexes 
                    WHERE tablename = 'key_sessions' AND indexname = 'idx_key_sessions_request_id'
                ) THEN
                    CREATE INDEX idx_key_sessions_request_id ON key_sessions(request_id);
                END IF;
            END$$;
        """))