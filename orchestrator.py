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
import subprocess

APP = FastAPI(title="Q-OPSEC Orchestrator", version="0.4.0")

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

def pidfile_path(name: str) -> Path:
    custom = service_cfg(name).get("pid_file")
    if custom:
        return (BASE_DIR / custom).resolve()
    logs_dir = Path(CONFIG["paths"]["logs_dir"]).resolve()
    return logs_dir / f"{name}.pid"

def write_pidfile(name: str, pid: int):
    p = pidfile_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(pid), encoding="utf-8")

def read_pidfile(name: str) -> Optional[int]:
    p = pidfile_path(name)
    if p.exists():
        try:
            return int(p.read_text(encoding="utf-8").strip())
        except Exception:
            return None
    return None

def is_pid_running(pid: int) -> bool:
    if pid is None:
        return False
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            return str(pid) in out.decode(errors="ignore")
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False

def kill_pid_windows(pid: int) -> bool:
    try:
        subprocess.check_call(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return True
    except Exception:
        return False

def find_pid_by_port_windows(port: int) -> Optional[int]:
    try:
        cmd = f'netstat -ano | findstr :{port}'
        out = subprocess.check_output(
            cmd, shell=True, creationflags=subprocess.CREATE_NO_WINDOW
        )
        lines = out.decode(errors="ignore").splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                return int(parts[-1])
    except Exception:
        return None
    return None

def find_pid_by_port_unix(port: int) -> Optional[int]:
    try:
        out = subprocess.check_output(["lsof", f"-i:{port}", "-sTCP:LISTEN", "-Pn"])
        lines = out.decode(errors="ignore").splitlines()
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    except Exception:
        pass
    try:
        out = subprocess.check_output(["fuser", "-n", "tcp", str(port)])
        txt = out.decode(errors="ignore").strip()
        for token in txt.split():
            if token.isdigit():
                return int(token)
    except Exception:
        pass
    return None

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

async def start_service(name: str) -> Dict[str, Any]:
    cfg = service_cfg(name)
    # Já rodando com handle?
    if name in STATE and STATE[name].get("proc") and getattr(STATE[name]["proc"], "poll", lambda: None)() is None:
        return {"status": "running", "pid": STATE[name]["proc"].pid}

    # Já rodando por PID file?
    exist_pid = read_pidfile(name)
    if exist_pid and is_pid_running(exist_pid):
        STATE[name] = {
            "proc": None,
            "pid": exist_pid,
            "started_at": None,
            "log_file": cfg.get("log_file"),
            "health": cfg.get("health"),
            "popen": True
        }
        return {"status": "running", "pid": exist_pid}

    cwd = (BASE_DIR / cfg["cwd"]).resolve() if cfg.get("cwd") else BASE_DIR
    cmd = cfg["start"]
    logs_dir = Path(CONFIG["paths"]["logs_dir"]).resolve()
    log_file = (BASE_DIR / (cfg.get("log_file") or (logs_dir / f"{name}.log"))).as_posix()

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    # No Windows, 'buffering=0' é inválido em texto — abre em binário para safe append
    stdout = open(log_file, "ab")
    stderr = stdout
    env = env_for_service(cfg)

    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd.as_posix(),
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                shell=False
            )
        except FileNotFoundError:
            # Para comandos .cmd/.bat no PATH
            proc = subprocess.Popen(
                " ".join(cmd),
                cwd=cwd.as_posix(),
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                shell=True
            )
        write_pidfile(name, proc.pid)
        STATE[name] = {
            "proc": proc,
            "pid": proc.pid,
            "started_at": time.time(),
            "log_file": log_file,
            "health": cfg.get("health"),
            "popen": True
        }
        return {"status": "started", "pid": proc.pid, "log_file": log_file}

    # Unix-like: prefer asyncio
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd.as_posix(),
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        write_pidfile(name, proc.pid)
        STATE[name] = {
            "proc": proc,
            "pid": proc.pid,
            "started_at": time.time(),
            "log_file": log_file,
            "health": cfg.get("health"),
            "popen": False
        }
        return {"status": "started", "pid": proc.pid, "log_file": log_file}
    except NotImplementedError:
        # fallback raro
        proc = subprocess.Popen(
            cmd,
            cwd=cwd.as_posix(),
            env=env,
            stdout=stdout,
            stderr=stderr,
            shell=False
        )
        write_pidfile(name, proc.pid)
        STATE[name] = {
            "proc": proc,
            "pid": proc.pid,
            "started_at": time.time(),
            "log_file": log_file,
            "health": cfg.get("health"),
            "popen": True
        }
        return {"status": "started", "pid": proc.pid, "log_file": log_file}

async def graceful_shutdown_if_possible(cfg: Dict[str, Any]) -> None:
    shutdown_url = None
    if cfg.get("health") and "actuator/health" in cfg["health"]:
        shutdown_url = cfg["health"].replace("/actuator/health", "/actuator/shutdown")
    if not shutdown_url:
        return
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(shutdown_url)
        await asyncio.sleep(1.5)
    except Exception:
        pass

async def stop_service(name: str, timeout: float = 12.0) -> Dict[str, Any]:
    cfg = service_cfg(name)
    state = STATE.get(name, {})
    proc = state.get("proc")
    pid = state.get("pid")
    port = cfg.get("port")

    # 1) Graceful shutdown via Actuator (se houver)
    await graceful_shutdown_if_possible(cfg)

    # 2) Descobrir PID: STATE -> pidfile -> porta
    if not pid:
        pid = read_pidfile(name)

    # Se Windows e com mvn/mvnw que spawna java.exe, descubra por porta
    if not pid and port:
        if sys.platform.startswith("win"):
            pid = find_pid_by_port_windows(int(port))
        else:
            pid = find_pid_by_port_unix(int(port))

    # Sem PID e sem porta => nada a fazer
    if not pid and not port:
        return {"status": "not_running"}

    # 3) Se temos handle Popen/asyncio, tenta parar via handle
    if proc is not None:
        # Tenta terminate/kill no handle
        if hasattr(proc, "terminate"):
            try:
                proc.terminate()
            except Exception:
                pass
            # Espera sair
            start_t = time.time()
            while (time.time() - start_t) < timeout:
                try:
                    if proc.poll() is not None:
                        break
                except AttributeError:
                    # asyncio subprocess
                    if proc.returncode is not None:
                        break
                await asyncio.sleep(0.2)
            # Se ainda vivo, kill
            try:
                # Popen tem kill(); asyncio também
                proc.kill()
            except Exception:
                pass
        # Continua para checar porta/PID e limpar pidfile

    # 4) Kill por PID (se ainda estiver vivo)
    if pid and is_pid_running(pid):
        if sys.platform.startswith("win"):
            kill_pid_windows(pid)
        else:
            try:
                os.kill(pid, signal.SIGTERM)
                start_t = time.time()
                while (time.time() - start_t) < timeout and is_pid_running(pid):
                    await asyncio.sleep(0.2)
                if is_pid_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    # 5) Se ainda há alguém na porta, mate o PID da porta (multi-processo, wrapper)
    if port:
        end_time = time.time() + timeout
        while time.time() < end_time:
            port_pid = None
            if sys.platform.startswith("win"):
                port_pid = find_pid_by_port_windows(int(port))
            else:
                port_pid = find_pid_by_port_unix(int(port))
            if port_pid is None:
                break
            if pid is None or port_pid != pid:
                # Kill do processo real que está na porta
                if sys.platform.startswith("win"):
                    kill_pid_windows(port_pid)
                else:
                    try:
                        os.kill(port_pid, signal.SIGTERM)
                        await asyncio.sleep(0.5)
                        if find_pid_by_port_unix(int(port)) is not None:
                            os.kill(port_pid, signal.SIGKILL)
                    except Exception:
                        pass
            await asyncio.sleep(0.5)

    # 6) Limpa pidfile se processo caiu
    try:
        pf = pidfile_path(name)
        if not pid or not is_pid_running(pid):
            pf.unlink(missing_ok=True)
    except Exception:
        pass

    # 7) Status final: confirma que porta liberou e PID não existe
    still_running = bool(pid and is_pid_running(pid))
    still_on_port = False
    if port:
        if sys.platform.startswith("win"):
            still_on_port = find_pid_by_port_windows(int(port)) is not None
        else:
            still_on_port = find_pid_by_port_unix(int(port)) is not None

    if not still_running and not still_on_port:
        return {"status": "stopped", "pid": pid}
    if not still_running and still_on_port:
        return {"status": "killed", "pid": pid}
    return {"status": "not_running", "pid": pid}

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

@APP.on_event("startup")
def _startup():
    load_config()
    for name, cfg in CONFIG.get("services", {}).items():
        pid = read_pidfile(name)
        if pid and is_pid_running(pid):
            STATE[name] = {
                "proc": None,      
                "pid": pid,
                "started_at": None,
                "log_file": cfg.get("log_file"),
                "health": cfg.get("health"),
                "popen": True
            }

@APP.get("/services")
def list_services():
    out = []
    for name, cfg in CONFIG.get("services", {}).items():
        state = STATE.get(name, {})
        pid = state.get("pid") or read_pidfile(name)
        running = bool(state.get("proc") and getattr(state["proc"], "poll", lambda: None)() is None) or (pid and is_pid_running(pid))
        out.append({
            "name": name,
            "pid": pid,
            "running": running,
            "started_at": state.get("started_at"),
            "health": cfg.get("health"),
            "log_file": state.get("log_file") or cfg.get("log_file"),
            "type": cfg.get("type"),
            "port": cfg.get("port"),
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
    pid = state.get("pid") or read_pidfile(name)
    running = bool(state.get("proc") and getattr(state["proc"], "poll", lambda: None)() is None) or (pid and is_pid_running(pid))
    health_url = cfg.get("health")
    ok, code, text = await check_health(health_url) if health_url else (False, None, None)
    return {
        "name": name,
        "running": running,
        "pid": pid,
        "health_ok": ok,
        "health_code": code,
        "health_sample": text,
    }

@APP.get("/services/{name}/logs")
def api_logs(name: str, lines: int = Query(200, ge=1, le=2000)):
    cfg = service_cfg(name)
    log_file = (STATE.get(name, {}).get("log_file")
                or cfg.get("log_file")
                or (Path(CONFIG["paths"]["logs_dir"]) / f"{name}.log").as_posix())
    return {"name": name, "log_file": log_file, "tail": tail_log(log_file, lines)}

class RequestSpec(BaseModel):
    method: str = Field(..., pattern="(?i)^(GET|POST|PUT|DELETE|PATCH)$")
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    json: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    timeout: float = 15.0

@APP.post("/request")
async def api_request(spec: RequestSpec):
    try:
        async with httpx.AsyncClient(timeout=spec.timeout) as client:
            r = await client.request(spec.method.upper(), spec.url, headers=spec.headers, json=spec.json, params=spec.params)
        return {"status_code": r.status_code, "headers": dict(r.headers), "body": r.text[:4000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@APP.post("/start/all")
async def start_all():
    order = CONFIG.get("startup_order", list(CONFIG.get("services", {}).keys()))
    results = []
    for name in order:
        results.append({name: await start_service(name)})
        await asyncio.sleep(1.0)
    return results

@APP.post("/stop/all")
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
        "X-API-Key": api_key or os.environ.get("CLASSIFY_API_KEY", service_cfg("classification_agent").get("env", {}).get("CLASSIFY_API_KEY", "your-api-key-for-authentication")),
    }
    url = "http://127.0.0.1:8088/api/v1/predict"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=payload)
        return {"status_code": r.status_code, "body": r.text[:4000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import re
from datetime import datetime
from collections import defaultdict

# ============== TRACE & FLOW ENDPOINTS ==============

def search_in_log(log_path: str, request_id: str, context_lines: int = 2) -> List[Dict[str, Any]]:
    """Busca request_id no log e retorna linhas com contexto"""
    p = Path(log_path)
    if not p.exists():
        return []
    
    matches = []
    try:
        lines = p.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if request_id in line:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                matches.append({
                    "line_number": i + 1,
                    "line": line,
                    "context": lines[start:end],
                    "timestamp": extract_timestamp(line)
                })
    except Exception as e:
        return [{"error": str(e)}]
    return matches

def extract_timestamp(line: str) -> Optional[str]:
    """Tenta extrair timestamp de uma linha de log"""
    # Padrões comuns: ISO8601, logs Spring, etc.
    patterns = [
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?',  # ISO8601
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',                # DD/MM/YYYY HH:MM:SS
        r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]',            # [YYYY-MM-DD HH:MM:SS]
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(0)
    return None

@APP.get("/trace/{request_id}")
async def trace_request(request_id: str, context_lines: int = Query(2, ge=0, le=10)):
    """Rastreia request_id em todos os logs dos serviços"""
    results = {}
    for name, cfg in CONFIG.get("services", {}).items():
        log_file = cfg.get("log_file") or (Path(CONFIG["paths"]["logs_dir"]) / f"{name}.log").as_posix()
        matches = search_in_log(log_file, request_id, context_lines)
        if matches:
            results[name] = {
                "service": name,
                "log_file": log_file,
                "matches": matches,
                "count": len(matches)
            }
    
    if not results:
        return {"request_id": request_id, "status": "not_found", "services": {}}
    
    return {
        "request_id": request_id,
        "status": "found",
        "total_matches": sum(r["count"] for r in results.values()),
        "services": results
    }

@APP.get("/flow/{request_id}")
async def flow_status(request_id: str):
    """Mostra o estado atual da requisição no fluxo (pipeline)"""
    # Pipeline esperado (ajuste conforme sua arquitetura)
    pipeline = [
        {"service": "interceptor_api", "endpoint": "/intercept", "port": 8080},
        {"service": "context_api", "endpoint": "/context/assemble", "port": 8081},
        {"service": "risk_service", "endpoint": "/assess", "port": 8082},
        {"service": "confiability_service", "endpoint": "/classify", "port": 8083},
        {"service": "classification_agent", "endpoint": "/api/v1/predict", "port": 8088},
        {"service": "rl_engine", "endpoint": "/act", "port": 9009},
        {"service": "handshake_negotiator", "endpoint": "/handshake", "port": 8001},
        {"service": "kms_service", "endpoint": "/keys", "port": 8002},
        {"service": "key_destination_engine", "endpoint": "/deliver", "port": 8003},
        {"service": "crypto_module", "endpoint": "/encrypt", "port": 8004},
        {"service": "validation_send_api", "endpoint": "/validation/send", "port": 8005},
    ]
    
    flow_state = []
    for step in pipeline:
        name = step["service"]
        cfg = CONFIG.get("services", {}).get(name)
        if not cfg:
            continue
        
        log_file = cfg.get("log_file") or (Path(CONFIG["paths"]["logs_dir"]) / f"{name}.log").as_posix()
        matches = search_in_log(log_file, request_id, context_lines=0)
        
        status = "pending"
        last_seen = None
        error = None
        
        if matches:
            status = "processed"
            last_seen = matches[-1].get("timestamp")
            # Detecta erro
            for m in matches:
                if any(kw in m["line"].lower() for kw in ["error", "exception", "failed", "400", "500"]):
                    status = "error"
                    error = m["line"][:200]
                    break
        
        flow_state.append({
            "step": len(flow_state) + 1,
            "service": name,
            "endpoint": step["endpoint"],
            "status": status,
            "last_seen": last_seen,
            "error": error,
            "matches_count": len(matches)
        })
    
    # Determina etapa atual
    current_step = None
    for i, step in enumerate(flow_state):
        if step["status"] == "error":
            current_step = i + 1
            break
        if step["status"] == "pending":
            current_step = i + 1
            break
    if current_step is None and flow_state:
        current_step = len(flow_state)  # completou
    
    return {
        "request_id": request_id,
        "current_step": current_step,
        "total_steps": len(flow_state),
        "flow": flow_state
    }

@APP.get("/timeline/{request_id}")
async def timeline(request_id: str):
    """Monta timeline cronológica da requisição"""
    events = []
    
    for name, cfg in CONFIG.get("services", {}).items():
        log_file = cfg.get("log_file") or (Path(CONFIG["paths"]["logs_dir"]) / f"{name}.log").as_posix()
        matches = search_in_log(log_file, request_id, context_lines=0)
        
        for m in matches:
            ts = m.get("timestamp")
            events.append({
                "timestamp": ts,
                "service": name,
                "line": m["line"],
                "line_number": m["line_number"]
            })
    
    # Ordena por timestamp
    events_sorted = sorted(
        [e for e in events if e["timestamp"]],
        key=lambda x: x["timestamp"]
    )
    
    return {
        "request_id": request_id,
        "total_events": len(events_sorted),
        "timeline": events_sorted
    }

@APP.get("/requests/active")
async def active_requests():
    """Lista request_ids ativos nos últimos N minutos (heurística)"""
    request_ids = set()
    pattern = re.compile(r'req[_-][\w\d]+')  # ajuste conforme seu padrão de request_id
    
    for name, cfg in CONFIG.get("services", {}).items():
        log_file = cfg.get("log_file") or (Path(CONFIG["paths"]["logs_dir"]) / f"{name}.log").as_posix()
        recent = tail_log(log_file, lines=500)
        for line in recent:
            matches = pattern.findall(line)
            request_ids.update(matches)
    
    return {
        "active_request_ids": sorted(list(request_ids)),
        "count": len(request_ids)
    }

if __name__ == "__main__":
    uvicorn.run("orchestrator:APP", host="0.0.0.0", port=8090, reload=False)