from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional

Base = declarative_base()

class KeySession(Base):
    __tablename__ = "key_sessions"
    session_id = Column(String, primary_key=True, index=True)
    algorithm = Column(String, nullable=False)
    key_material = Column(String, nullable=False)  # em base64
    expires_at = Column(DateTime, nullable=False)

class KeyResponse(BaseModel):
    session_id: str
    algorithm: str
    key_material: str
    expires_at: datetime

class CreateKeyRequest(BaseModel):
    session_id: Optional[str] = None
    algorithm: str
    ttl_seconds: int = 3600
    strict: bool = False

class CreateKeyResponse(BaseModel):
    session_id: str
    requested_algorithm: str
    selected_algorithm: str
    key_material: str
    expires_at: datetime          # <-- aqui: datetime
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    source_of_key: str  # "qkd" | "pqc" | "classical"