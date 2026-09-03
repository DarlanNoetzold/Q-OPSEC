from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/kms_db"

engine = create_engine(DATABASE_URL, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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