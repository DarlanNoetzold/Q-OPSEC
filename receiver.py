# receiver.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Mock Key Receiver")

from fastapi import FastAPI, Request
app = FastAPI()
@app.post("/receiver")
async def receiver(req: Request):
    data = await req.json()
    return {"ok": True, "received": list(data.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("receiver:app", host="0.0.0.0", port=9000, reload=True)