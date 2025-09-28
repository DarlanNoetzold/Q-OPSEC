import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any
from service import RLEngineService
from negotiator_client import send_to_handshake

class Settings:
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    log_level: str = "info"
    registry_path: str = "./rl_registry.json"
    handshake_url: str = "http://localhost:8001/handshake"

settings = Settings()
app = FastAPI(title="RL Engine Service")
rl_service = RLEngineService(Path(settings.registry_path))

class ContextRequest(BaseModel):
    request_id: str = Field(...)
    source: str = Field(...)
    destination: str = Field(...)
    security_level: str = Field(...)
    risk_score: float = Field(...)
    conf_score: float = Field(...)
    dst_props: Dict[str, Any] = Field(default_factory=dict)

@app.post("/act")
def act(req: ContextRequest):
    payload = rl_service.build_negotiation_payload(req.dict())
    result = send_to_handshake(settings.handshake_url, payload)
    return {"negotiation": result, "payload": payload}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower()
    )