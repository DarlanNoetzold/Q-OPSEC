from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Ajuste os parâmetros abaixo (usuário, senha, IP, porta, DB)
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/kms_db"

# Criar engine
engine = create_engine(DATABASE_URL)

# Criar SessionLocal
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base ORM
Base = declarative_base()

# Dependência FastAPI para injetar sessão
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()