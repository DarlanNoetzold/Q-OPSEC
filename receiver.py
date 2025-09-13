# receiver.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Mock Key Receiver")

class Incoming(BaseModel):
    session_id: str
    algorithm: str
    key_material: str
    expires_at: str
    delivery_id: str | None = None
    metadata: dict | None = None

@app.post("/receive_key")
async def receive_key(body: Incoming, req: Request):
    print("[Receiver] Got key:", body.dict())
    return {
        "status": "ok",
        "received_at": datetime.utcnow().isoformat(),
        "peer": req.client.host
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("receiver:app", host="0.0.0.0", port=9000, reload=True)