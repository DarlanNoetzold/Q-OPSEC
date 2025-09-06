from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class KeySession(Base):
    __tablename__ = "key_sessions"
    session_id = Column(String, primary_key=True, index=True)
    algorithm = Column(String, nullable=False)
    key_material = Column(String, nullable=False)  # em base64
    expires_at = Column(DateTime, nullable=False)

# Request/Response via API
class CreateKeyRequest(BaseModel):
    session_id: str
    algorithm: str
    ttl_seconds: int = 300

class KeyResponse(BaseModel):
    session_id: str
    algorithm: str
    key_material: str
    expires_at: datetime