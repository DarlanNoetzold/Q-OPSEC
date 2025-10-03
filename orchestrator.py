import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field

APP = FastAPI(title="Q-OPSEC Orchestrator", version="0.2.1")

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "services.yaml"
STATE: Dict[str, Dict[str, Any]] = {}
CONFIG: Dict[str, Any] = {}

def load_config():
    global CONFIG
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {CONFIG_PATH}")
    CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    Path(CONFIG["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

def service_cfg(name: str) -> Dict[str, Any]:
    services = CONFIG.get("services", {})
    if name not in services:
        raise HTTPException(status_code=404, detail=f"Unknown service: {name}")
    return services[name]

def env_for_service(cfg: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    return env

async def start_service(name: str) -> Dict[str, Any]:
    cfg = service_cfg(name)
    if name in STATE and STATE[name].get("proc") and STATE[name]["proc"].poll() is None:
        return {"status": "running", "pid": STATE[name]["proc"].pid}

    cwd = (BASE_DIR / cfg["cwd"]).resolve() if cfg.get("cwd") else BASE_DIR
    cmd = cfg["start"]
    logs_dir = Path(CONFIG["paths"]["logs_dir"]).resolve()
    log_file = (logs_dir / f"{name}.log").as_posix()

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    stdout = open(log_file, "ab", buffering=0)
    stderr = stdout

    env = env_for_service(cfg)

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = 0x00000010  # CREATE_NEW_CONSOLE

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd.as_posix(),
        env=env,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags if sys.platform.startswith("win") else 0,
    )

    STATE[name] = {
        "proc": proc,
        "pid": proc.pid,
        "started_at": time.time(),
        "log_file": log_file,
        "health": cfg.get("health"),
    }
    return {"status": "started", "pid": proc.pid, "log_file": log_file}

async def stop_service(name: str, timeout: float = 8.0) -> Dict[str, Any]:
    if name not in STATE or STATE[name].get("proc") is None:
        return {"status": "not_running"}
    proc = STATE[name]["proc"]
    pid = STATE[name]["pid"]
    if proc.poll() is not None:
        return {"status": "not_running"}

    try:
        if sys.platform.startswith("win"):
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        return {"status": "stopped", "pid": pid}
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"status": "killed", "pid": pid}

async def restart_service(name: str) -> Dict[str, Any]:
    await stop_service(name)
    return await start_service(name)

async def check_health(url: str, timeout: float = 6.0) -> Tuple[bool, Optional[int], Optional[str]]:
    if not url:
        return (False, None, "no health url")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
        return (r.status_code == 200, r.status_code, r.text[:200])
    except Exception as e:
        return (False, None, str(e))

def tail_log(path: str, lines: int = 200) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            block_size = 1024
            data = b""
            while len(data.splitlines()) <= lines and file_size > 0:
                read_size = min(block_size, file_size)
                file_size -= read_size
                f.seek(file_size)
                data = f.read(read_size) + data
            return data.decode(errors="replace").splitlines()[-lines:]
        except Exception:
            return p.read_text(errors="replace").splitlines()[-lines:]

class RequestSpec(BaseModel):
    method: str = Field(..., pattern="(?i)^(GET|POST|PUT|DELETE|PATCH)$")
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    json: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    timeout: float = 15.0

@APP.on_event("startup")
def _startup():
    load_config()

@APP.get("/services")
def list_services():
    out = []
    for name, cfg in CONFIG.get("services", {}).items():
        state = STATE.get(name, {})
        out.append({
            "name": name,
            "pid": state.get("pid"),
            "running": bool(state.get("proc") and state["proc"].poll() is None),
            "started_at": state.get("started_at"),
            "health": cfg.get("health"),
            "log_file": state.get("log_file") or cfg.get("log_file"),
            "type": cfg.get("type"),
        })
    return out

@APP.post("/services/{name}/start")
async def api_start(name: str):
    return await start_service(name)

@APP.post("/services/{name}/stop")
async def api_stop(name: str):
    return await stop_service(name)

@APP.post("/services/{name}/restart")
async def api_restart(name: str):
    return await restart_service(name)

@APP.get("/services/{name}/status")
async def api_status(name: str):
    state = STATE.get(name, {})
    cfg = service_cfg(name)
    running = state.get("proc") and state["proc"].poll() is None
    health_url = cfg.get("health")
    ok, code, text = await check_health(health_url) if health_url else (False, None, None)
    return {
        "name": name,
        "running": bool(running),
        "pid": state.get("pid"),
        "health_ok": ok,
        "health_code": code,
        "health_sample": text,
    }

@APP.get("/services/{name}/logs")
def api_logs(name: str, lines: int = Query(200, ge=1, le=2000)):
    log_file = STATE.get(name, {}).get("log_file") or service_cfg(name).get("log_file")
    if not log_file:
        raise HTTPException(status_code=404, detail="No log_file configured")
    return {"name": name, "log_file": log_file, "tail": tail_log(log_file, lines)}

@APP.post("/request")
async def api_request(spec: RequestSpec):
    try:
        async with httpx.AsyncClient(timeout=spec.timeout) as client:
            r = await client.request(spec.method.upper(), spec.url, headers=spec.headers, json=spec.json, params=spec.params)
        return {"status_code": r.status_code, "headers": dict(r.headers), "body": r.text[:4000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@APP.post("/services/all/start")
async def start_all():
    order = CONFIG.get("startup_order", list(CONFIG.get("services", {}).keys()))
    results = []
    for name in order:
        results.append({name: await start_service(name)})
        await asyncio.sleep(1.0)
    return results

@APP.post("/services/all/stop")
async def stop_all():
    names = list(CONFIG.get("services", {}).keys())
    results = []
    for name in reversed(names):
        results.append({name: await stop_service(name)})
    return results

@APP.post("/demo/predict")
async def demo_predict(api_key: Optional[str] = None):
    payload = {
        "send_to_rl": True,
        "data": {
            "request_id_resolved": "req_123",
            "created_at": "2025-09-24T09:44:40.438788",
            "risk_score": 0.11,
            "conf_score": 0.085,
            "combined_score": 0.117,
            "risk_level": "Low",
            "conf_classification": "confidential",
            "src_geo": "EU",
            "src_device_type": "iot",
            "dst_service_type": "web",
            "dst_security_policy": "high",
            "src_mfa_status_norm": "disabled"
        }
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key or os.environ.get("CLASSIFY_API_KEY", "your-api-key-for-authentication"),
    }
    url = "http://127.0.0.1:8088/api/v1/predict"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=payload)
        return {"status_code": r.status_code, "body": r.text[:4000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("orchestrator:APP", host="0.0.0.0", port=8090, reload=True)