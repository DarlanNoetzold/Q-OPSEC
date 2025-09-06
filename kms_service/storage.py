import aioredis
from sqlalchemy.orm import Session
from models import KeySession
from datetime import datetime

# Redis cache
redis = aioredis.from_url("redis://localhost")

def save_session(db: Session, session_id, algorithm, key_material, expires_at):
    sess = KeySession(session_id=session_id, algorithm=algorithm,
                      key_material=key_material, expires_at=expires_at)
    db.add(sess)
    db.commit()
    # cache em Redis
    redis.setex(session_id, int((expires_at - datetime.utcnow()).total_seconds()), key_material)

def get_session(db: Session, session_id):
    # primeiro tenta Redis
    val = redis.get(session_id)
    if val:
        return val
    # se não, consulta db
    sess = db.query(KeySession).filter(KeySession.session_id == session_id).first()
    return sess