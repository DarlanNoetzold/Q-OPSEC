import asyncio
from datetime import datetime
import json
import base64
import os
import signal
import sys
import time
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Body, Query, Request
from pydantic import BaseModel, Field
import subprocess

APP = FastAPI(title="Q-OPSEC Orchestrator", version="0.6.0-linux-docker")

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "services.yaml"
STATE: Dict[str, Dict[str, Any]] = {}
CONFIG: Dict[str, Any] = {}

from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_swagger_url(cfg: Dict[str, Any], name: str) -> Optional[str]:
    port = cfg.get("port")
    if not port:
        return None
    
    service_type = cfg.get("type", "process")
    
    if name == "confiability_service":
        return f"http://192.168.18.18:{port}/swagger-ui"
    
    if service_type in ["spring", "java"]:
        return f"http://192.168.18.18:{port}/swagger-ui/index.html"
    
    if service_type in ["fastapi", "python", "process"]:
        start_cmd = cfg.get("start", [])
        if start_cmd and isinstance(start_cmd, list):
            if any("python" in str(cmd).lower() or "uvicorn" in str(cmd).lower() for cmd in start_cmd):
                return f"http://192.168.18.18:{port}/docs"
    
    return None

def docker_available() -> bool:
    """Check if Docker is available"""
    return shutil.which("docker") is not None

def docker_running() -> bool:
    if not docker_available():
        return False
    try:
        subprocess.check_output(
            ["docker", "info"],
            stderr=subprocess.DEVNULL,
            timeout=3
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False

def docker_container_exists(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            stderr=subprocess.DEVNULL
        )
        return name in out.decode().strip()
    except subprocess.CalledProcessError:
        return False

def docker_container_running(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            stderr=subprocess.DEVNULL
        )
        return name in out.decode().strip()
    except subprocess.CalledProcessError:
        return False

def docker_get_container_id(name: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.ID}}"],
            stderr=subprocess.DEVNULL
        )
        container_id = out.decode().strip()
        return container_id if container_id else None
    except subprocess.CalledProcessError:
        return None

def docker_image_exists(image: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "images", "-q", image],
            stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except subprocess.CalledProcessError:
        return False

async def docker_pull_image(image: str) -> Dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "pull", image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            return {"status": "pulled", "image": image}
        else:
            return {"status": "error", "error": stderr.decode()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def docker_start_container(cfg: Dict[str, Any]) -> Dict[str, Any]:
    name = cfg.get("container_name", cfg.get("name"))
    image = cfg.get("image")

    if not image:
        return {"status": "error", "error": "No image specified"}

    if docker_container_exists(name):
        if docker_container_running(name):
            container_id = docker_get_container_id(name)
            return {"status": "running", "container_id": container_id, "container_name": name}
        else:
            try:
                subprocess.check_call(
                    ["docker", "start", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                container_id = docker_get_container_id(name)
                return {"status": "started", "container_id": container_id, "container_name": name}
            except subprocess.CalledProcessError as e:
                return {"status": "error", "error": str(e)}

    if not docker_image_exists(image):
        pull_result = await docker_pull_image(image)
        if pull_result["status"] != "pulled":
            return pull_result

    cmd = ["docker", "run"]

    cmd.extend(["--name", name])

    if cfg.get("hostname"):
        cmd.extend(["--hostname", cfg["hostname"]])

    for key, value in cfg.get("env", {}).items():
        cmd.extend(["--env", f"{key}={value}"])

    for volume in cfg.get("volumes", []):
        cmd.extend(["--volume", volume])

    for port in cfg.get("ports", []):
        cmd.extend(["-p", port])

    if cfg.get("network"):
        cmd.extend(["--network", cfg["network"]])

    if cfg.get("workdir"):
        cmd.extend(["--workdir", cfg["workdir"]])

    restart = cfg.get("restart", "no")
    cmd.extend(["--restart", restart])

    if cfg.get("runtime"):
        cmd.extend(["--runtime", cfg["runtime"]])

    for label in cfg.get("labels", []):
        cmd.extend(["--label", label])

    for arg in cfg.get("extra_args", []):
        cmd.append(arg)

    cmd.append("-d")

    cmd.append(image)

    if cfg.get("command"):
        if isinstance(cfg["command"], list):
            cmd.extend(cfg["command"])
        else:
            cmd.append(cfg["command"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            container_id = stdout.decode().strip()
            return {
                "status": "started",
                "container_id": container_id,
                "container_name": name,
                "image": image
            }
        else:
            return {"status": "error", "error": stderr.decode()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def docker_stop_container(name: str, timeout: int = 10) -> Dict[str, Any]:
    if not docker_container_exists(name):
        return {"status": "not_found"}

    if not docker_container_running(name):
        return {"status": "not_running"}

    try:
        subprocess.check_call(
            ["docker", "stop", "-t", str(timeout), name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return {"status": "stopped", "container_name": name}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": str(e)}

async def docker_remove_container(name: str, force: bool = False) -> Dict[str, Any]:
    if not docker_container_exists(name):
        return {"status": "not_found"}

    cmd = ["docker", "rm"]
    if force:
        cmd.append("-f")
    cmd.append(name)

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"status": "removed", "container_name": name}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": str(e)}

def docker_get_logs(name: str, lines: int = 200) -> List[str]:
    if not docker_container_exists(name):
        return []

    try:
        out = subprocess.check_output(
            ["docker", "logs", "--tail", str(lines), name],
            stderr=subprocess.STDOUT
        )
        return out.decode(errors="replace").splitlines()
    except subprocess.CalledProcessError:
        return []

def docker_container_stats(name: str) -> Optional[Dict[str, Any]]:
    if not docker_container_running(name):
        return None

    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
            stderr=subprocess.DEVNULL
        )
        return json.loads(out.decode())
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


@APP.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = BASE_DIR / "dashboard.html"
    if not html_path.exists():
        return "<h1>Dashboard not found</h1><p>Place dashboard.html in the same directory</p>"
    return html_path.read_text(encoding="utf-8")

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
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def find_pid_by_port(port: int) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["lsof", f"-i:{port}", "-sTCP:LISTEN", "-Pn", "-t"],
            stderr=subprocess.DEVNULL
        )
        pids = out.decode(errors="ignore").strip().split()
        if pids and pids[0].isdigit():
            return int(pids[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        out = subprocess.check_output(
            ["fuser", "-n", "tcp", str(port)],
            stderr=subprocess.DEVNULL
        )
        txt = out.decode(errors="ignore").strip()
        for token in txt.split():
            if token.isdigit():
                return int(token)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL
        )
        lines = out.decode(errors="ignore").splitlines()
        for line in lines[1:]:  # Skip header
            if f":{port}" in line and "pid=" in line:
                import re
                match = re.search(r'pid=(\d+)', line)
                if match:
                    return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None

def tail_log(path: str, lines: int = 200) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []

    try:
        out = subprocess.check_output(
            ["tail", f"-n{lines}", str(p)],
            stderr=subprocess.DEVNULL
        )
        return out.decode(errors="replace").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
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

async def start_service(name: str) -> Dict[str, Any]:
    cfg = service_cfg(name)
    service_type = cfg.get("type", "process")

    if service_type == "docker":
        if not docker_running():
            return {"status": "error", "error": "Docker daemon not running"}
        return await docker_start_container(cfg)

    if name in STATE and STATE[name].get("proc"):
        proc = STATE[name]["proc"]
        if hasattr(proc, "returncode") and proc.returncode is None:
            return {"status": "running", "pid": proc.pid}

    exist_pid = read_pidfile(name)
    if exist_pid and is_pid_running(exist_pid):
        STATE[name] = {
            "proc": None,
            "pid": exist_pid,
            "started_at": None,
            "log_file": cfg.get("log_file"),
            "health": cfg.get("health"),
        }
        return {"status": "running", "pid": exist_pid}

    if service_type == "process" and cmd and isinstance(cmd, list) and len(cmd) > 0 and cmd[0].endswith("python"):
        req_file = cwd / "requirements.txt"
        if req_file.exists():
            try:
                print(f"Installing requirements for {name}...")
                subprocess.run([cmd[0], "-m", "pip", "install", "-r", str(req_file)], check=True)
            except Exception as e:
                print(f"Failed to install requirements for {name}: {e}")

    cwd = (BASE_DIR / cfg["cwd"]).resolve() if cfg.get("cwd") else BASE_DIR
    cmd = cfg["start"]
    logs_dir = Path(CONFIG["paths"]["logs_dir"]).resolve()
    log_file = (BASE_DIR / (cfg.get("log_file") or (logs_dir / f"{name}.log"))).as_posix()

    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    stdout = open(log_file, "ab", buffering=0)
    stderr = stdout
    env = env_for_service(cfg)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True
        )

        write_pidfile(name, proc.pid)
        STATE[name] = {
            "proc": proc,
            "pid": proc.pid,
            "started_at": time.time(),
            "log_file": log_file,
            "health": cfg.get("health"),
        }
        return {"status": "started", "pid": proc.pid, "log_file": log_file}

    except Exception as e:
        return {"status": "error", "error": str(e)}

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
    service_type = cfg.get("type", "process")

    if service_type == "docker":
        container_name = cfg.get("container_name", name)
        return await docker_stop_container(container_name, int(timeout))

    state = STATE.get(name, {})
    proc = state.get("proc")
    pid = state.get("pid")
    port = cfg.get("port")

    await graceful_shutdown_if_possible(cfg)

    if not pid:
        pid = read_pidfile(name)

    if not pid and port:
        pid = find_pid_by_port(int(port))

    if not pid:
        return {"status": "not_running"}

    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass

        start_t = time.time()
        while (time.time() - start_t) < timeout:
            if proc.returncode is not None:
                break
            await asyncio.sleep(0.2)

        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass

    if pid and is_pid_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)

            start_t = time.time()
            while (time.time() - start_t) < timeout and is_pid_running(pid):
                await asyncio.sleep(0.2)

            if is_pid_running(pid):
                os.kill(pid, signal.SIGKILL)
                await asyncio.sleep(0.5)
        except Exception:
            pass

    if port:
        end_time = time.time() + timeout
        while time.time() < end_time:
            port_pid = find_pid_by_port(int(port))
            if port_pid is None:
                break

            if port_pid != pid:
                try:
                    os.kill(port_pid, signal.SIGTERM)
                    await asyncio.sleep(0.5)
                    if find_pid_by_port(int(port)) is not None:
                        os.kill(port_pid, signal.SIGKILL)
                except Exception:
                    pass

            await asyncio.sleep(0.5)

    try:
        pf = pidfile_path(name)
        if not pid or not is_pid_running(pid):
            pf.unlink(missing_ok=True)
    except Exception:
        pass

    still_running = bool(pid and is_pid_running(pid))
    still_on_port = bool(port and find_pid_by_port(int(port)) is not None)

    if not still_running and not still_on_port:
        STATE.pop(name, None)
        return {"status": "stopped", "pid": pid}
    if not still_running and still_on_port:
        return {"status": "killed", "pid": pid}
    return {"status": "failed_to_stop", "pid": pid}

async def restart_service(name: str) -> Dict[str, Any]:
    await stop_service(name)
    await asyncio.sleep(1.0)
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
        service_type = cfg.get("type", "process")

        if service_type == "docker":
            container_name = cfg.get("container_name", name)
            if docker_container_running(container_name):
                STATE[name] = {
                    "type": "docker",
                    "container_name": container_name,
                    "container_id": docker_get_container_id(container_name),
                    "running": True
                }
        else:
            pid = read_pidfile(name)
            if pid and is_pid_running(pid):
                STATE[name] = {
                    "proc": None,
                    "pid": pid,
                    "started_at": None,
                    "log_file": cfg.get("log_file"),
                    "health": cfg.get("health"),
                }

@APP.get("/docker/status")
def docker_status():
    return {
        "available": docker_available(),
        "running": docker_running()
    }

@APP.get("/services")
def list_services():
    out = []
    for name, cfg in CONFIG.get("services", {}).items():
        service_type = cfg.get("type", "process")
        state = STATE.get(name, {})

        if service_type == "docker":
            container_name = cfg.get("container_name", name)
            running = docker_container_running(container_name)
            container_id = docker_get_container_id(container_name) if running else None

            out.append({
                "name": name,
                "type": "docker",
                "container_name": container_name,
                "container_id": container_id,
                "image": cfg.get("image"),
                "running": running,
                "health": cfg.get("health"),
                "ports": cfg.get("ports", []),
                "swagger_url": get_swagger_url(cfg, name),
            })
        else:
            pid = state.get("pid") or read_pidfile(name)

            running = False
            if state.get("proc") and hasattr(state["proc"], "returncode"):
                running = state["proc"].returncode is None
            elif pid:
                running = is_pid_running(pid)

            out.append({
                "name": name,
                "type": "process",
                "pid": pid,
                "running": running,
                "started_at": state.get("started_at"),
                "health": cfg.get("health"),
                "log_file": state.get("log_file") or cfg.get("log_file"),
                "port": cfg.get("port"),
                "start": cfg.get("start"),
                "cwd": cfg.get("cwd"),
                "swagger_url": get_swagger_url(cfg, name),
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
    cfg = service_cfg(name)
    service_type = cfg.get("type", "process")

    if service_type == "docker":
        container_name = cfg.get("container_name", name)
        running = docker_container_running(container_name)
        container_id = docker_get_container_id(container_name)
        stats = docker_container_stats(container_name) if running else None

        health_url = cfg.get("health")
        ok, code, text = await check_health(health_url) if health_url else (False, None, None)

        return {
            "name": name,
            "type": "docker",
            "running": running,
            "container_id": container_id,
            "container_name": container_name,
            "stats": stats,
            "health_ok": ok,
            "health_code": code,
            "health_sample": text,
        }
    else:
        state = STATE.get(name, {})
        pid = state.get("pid") or read_pidfile(name)

        running = False
        if state.get("proc") and hasattr(state["proc"], "returncode"):
            running = state["proc"].returncode is None
        elif pid:
            running = is_pid_running(pid)

        health_url = cfg.get("health")
        ok, code, text = await check_health(health_url) if health_url else (False, None, None)

        return {
            "name": name,
            "type": "process",
            "running": running,
            "pid": pid,
            "health_ok": ok,
            "health_code": code,
            "health_sample": text,
        }

@APP.get("/services/{name}/logs")
def api_logs(name: str, lines: int = Query(200, ge=1, le=2000)):
    cfg = service_cfg(name)
    service_type = cfg.get("type", "process")

    if service_type == "docker":
        container_name = cfg.get("container_name", name)
        logs = docker_get_logs(container_name, lines)
        return {"name": name, "type": "docker", "container_name": container_name, "tail": logs}
    else:
        log_file = (STATE.get(name, {}).get("log_file")
                    or cfg.get("log_file")
                    or (Path(CONFIG["paths"]["logs_dir"]) / f"{name}.log").as_posix())
        return {"name": name, "type": "process", "log_file": log_file, "tail": tail_log(log_file, lines)}

@APP.post("/services/{name}/remove")
async def api_remove_container(name: str, force: bool = False):
    cfg = service_cfg(name)
    if cfg.get("type") != "docker":
        raise HTTPException(status_code=400, detail="Service is not a Docker container")

    container_name = cfg.get("container_name", name)
    return await docker_remove_container(container_name, force)

class RequestSpec(BaseModel):
    method: str = Field(..., pattern="(?i)^(GET|POST|PUT|DELETE|PATCH)$")
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = Field(default=None, alias="json")
    params: Dict[str, Any] = Field(default_factory=dict)
    timeout: float = 15.0

    class Config:
        populate_by_name = True

@APP.post("/request")
async def api_request(spec: RequestSpec):
    try:
        async with httpx.AsyncClient(timeout=spec.timeout) as client:
            r = await client.request(
                spec.method.upper(),
                spec.url,
                headers=spec.headers,
                json=spec.payload,
                params=spec.params
            )
            return {
                "status_code": r.status_code,
                "headers": dict(r.headers),
                "body": r.text[:4000]
            }
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
        "X-API-Key": api_key or os.environ.get(
            "CLASSIFY_API_KEY",
            service_cfg("classification_agent").get("env", {}).get(
                "CLASSIFY_API_KEY", "your-api-key-for-authentication"
            )
        ),
    }
    url = "http://127.0.0.1:8088/api/v1/predict"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            return {"status_code": r.status_code, "body": r.text[:4000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import re
from collections import defaultdict

def search_in_log(log_path: str, request_id: str, context_lines: int = 2) -> List[Dict[str, Any]]:
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
    patterns = [
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?',  # ISO8601
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # DD/MM/YYYY HH:MM:SS
        r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]',  # [YYYY-MM-DD HH:MM:SS]
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(0)
    return None

@APP.get("/trace/{request_id}")
async def trace_request(request_id: str, context_lines: int = Query(2, ge=0, le=10)):
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
    pipeline = [
        {"service": "interceptor_api", "endpoint": "/intercept", "port": 8080},
        {"service": "context_api", "endpoint": "/context/enrich", "port": 65534},
        {"service": "risk_service", "endpoint": "/predict/", "port": 8000},
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
            # Detect errors
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

    current_step = None
    for i, step in enumerate(flow_state):
        if step["status"] == "error":
            current_step = i + 1
            break
        if step["status"] == "pending":
            current_step = i + 1
            break
    if current_step is None and flow_state:
        current_step = len(flow_state)

    return {
        "request_id": request_id,
        "current_step": current_step,
        "total_steps": len(flow_state),
        "flow": flow_state
    }

@APP.get("/timeline/{request_id}")
async def timeline(request_id: str):
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
    request_ids = set()
    pattern = re.compile(r'req[_-][\w\d]+')

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


class ServiceConfigUpdate(BaseModel):
    start: Optional[List[str]] = None
    cwd: Optional[str] = None

@APP.put("/services/{name}/config")
async def update_service_config(name: str, config: ServiceConfigUpdate = Body(...)):
    if name not in CONFIG.get("services", {}):
        raise HTTPException(status_code=404, detail="Service not found")

    svc = CONFIG["services"][name]
    updated = False

    if config.start is not None:
        svc["start"] = config.start
        updated = True
    if config.cwd is not None:
        svc["cwd"] = config.cwd
        updated = True

    if updated:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(CONFIG, f)
        load_config()

    return {"status": "updated", "service": name, "config": svc}

def get_api_key_for_service(svc):
    if "env" in svc:
        key = svc["env"].get("CLASSIFY_API_KEY") or svc["env"].get("API_KEY")
        if key: return key
    return "your-api-key-for-authentication"

async def proxy_get(url, headers=None):
    headers = headers or {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

@APP.get("/metrics/{service_name}/sessions")
async def get_metrics_sessions(service_name: str):
    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    headers = {}
    api_key = get_api_key_for_service(svc)
    if api_key:
        headers["X-API-Key"] = api_key

    if service_name == "classification_agent":
        url = f"{base_url}/api/v1/training/sessions"
    elif service_name == "confiability_service":
        url = f"{base_url}/confidentiality/metrics/sessions"
    elif service_name == "risk_service":
        url = f"{base_url}/risk/metrics/sessions"
    else:
        raise HTTPException(status_code=400, detail="Unknown service for metrics")

    return await proxy_get(url, headers=headers)

@APP.get("/metrics/{service_name}/sessions/{session_id}")
async def get_metrics_session_detail(service_name: str, session_id: str):
    if not session_id or session_id == "undefined":
        raise HTTPException(status_code=400, detail="Invalid session_id")

    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    headers = {}
    api_key = get_api_key_for_service(svc)
    if api_key:
        headers["X-API-Key"] = api_key

    if service_name == "classification_agent":
        url = f"{base_url}/api/v1/training/{session_id}/summary"
    elif service_name == "confiability_service":
        url = f"{base_url}/confidentiality/metrics/{session_id}"
    elif service_name == "risk_service":
        url = f"{base_url}/risk/metrics/{session_id}"
    else:
        raise HTTPException(status_code=400, detail="Unknown service for metrics")

    return await proxy_get(url, headers=headers)

@APP.get("/metrics/{service_name}/images")
async def get_metrics_images(service_name: str):
    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    headers = {}
    api_key = get_api_key_for_service(svc)
    if api_key:
        headers["X-API-Key"] = api_key

    if service_name == "classification_agent":
        url = f"{base_url}/api/v1/training/images"
    elif service_name == "confiability_service":
        url = f"{base_url}/confidentiality/metrics/images"
    elif service_name == "risk_service":
        url = f"{base_url}/risk/metrics/images"
    else:
        raise HTTPException(status_code=400, detail="Unknown service for metrics images")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch images list")

@APP.get("/metrics/{service_name}/sessions/{session_id}/{image_name}")
async def get_metrics_image(service_name: str, session_id: str, image_name: str):
    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    local_path = Path(CONFIG["paths"]["logs_dir"]) / "metrics" / service_name / session_id / image_name
    if local_path.exists():
        return FileResponse(local_path)

    headers = {}
    api_key = get_api_key_for_service(svc)
    if api_key:
        headers["X-API-Key"] = api_key

    if service_name == "classification_agent":
        url = f"{base_url}/api/v1/training/{session_id}/images/{image_name}"
    elif service_name == "confiability_service":
        url = f"{base_url}/confidentiality/metrics/{session_id}/{image_name}"
    elif service_name == "risk_service":
        url = f"{base_url}/risk/metrics/{session_id}/{image_name}"
    else:
        raise HTTPException(status_code=400, detail="Unknown service for metrics images")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return StreamingResponse(resp.aiter_bytes(), media_type=resp.headers.get("content-type"))
        else:
            raise HTTPException(status_code=resp.status_code, detail=f"Image not found: {image_name}")

@APP.get("/datasets/{service_name}")
async def get_datasets(service_name: str):
    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    headers = {}
    api_key = get_api_key_for_service(svc)
    if api_key:
        headers["X-API-Key"] = api_key

    if service_name == "classification_agent":
        url = f"{base_url}/api/v1/datasets"
    elif service_name == "confiability_service":
        url = f"{base_url}/datasets"
    elif service_name == "risk_service":
        url = f"{base_url}/datasets"
    else:
        raise HTTPException(status_code=400, detail="Unknown service for datasets")

    return await proxy_get(url, headers=headers)

@APP.get("/datasets/{service_name}/{dataset_name}/preview")
async def get_dataset_preview(service_name: str, dataset_name: str, file: str, n: int = 20):
    if not file:
        raise HTTPException(status_code=400, detail="File parameter required")

    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    headers = {}
    api_key = get_api_key_for_service(svc)
    if api_key:
        headers["X-API-Key"] = api_key

    if service_name == "confiability_service":
        url = f"{base_url}/datasets/{dataset_name}/preview?file={file}&n={n}"
    elif service_name == "risk_service":
        url = f"{base_url}/datasets/{dataset_name}/preview?file={file}&n={n}"
    elif service_name == "classification_agent":
        url = f"{base_url}/api/v1/datasets/{dataset_name}/preview?file={file}&n={n}"
    else:
        raise HTTPException(status_code=400, detail="Unknown service for datasets")

    return await proxy_get(url, headers=headers)

@APP.post("/callback")
async def callback(request: Request):
    try:
        data = await request.json()
        return {"status": "success", "message": "Data successfully received at origin", "received_data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@APP.post("/run-pipeline")
async def run_pipeline(data: Dict[str, Any] = Body(...)):
    # 1. Identidade Única Profissional (PhD Traceability)
    request_id = data.get("request_id") or data.get("requestId") or f"req-{int(time.time())}-{uuid4().hex[:6]}"
    
    # 2. Inicialização do Payload com IDs Sincronizados
    current_data = data
    current_data["request_id"] = request_id
    current_data["requestId"] = request_id

    pipeline = [
        {"name": "interceptor_api", "port": 8080, "endpoint": "/intercept", "method": "POST"},
        {"name": "context_api", "port": 65534, "endpoint": "/context/enrich", "method": "POST"},
        {"name": "risk_service", "port": 8000, "endpoint": "/predict/", "method": "POST"},
        {"name": "classification_agent", "port": 8088, "endpoint": "/api/v1/predict", "method": "POST"},
        {"name": "rl_engine", "port": 9009, "endpoint": "/act", "method": "POST"},
        {"name": "handshake_negotiator", "port": 8001, "endpoint": "/handshake", "method": "POST"},
        {"name": "kms_service", "port": 8002, "endpoint": "/kms/create_key", "method": "POST"},
        {"name": "key_destination_engine", "port": 8003, "endpoint": "/deliver", "method": "POST"},
        {"name": "crypto_module", "port": 8004, "endpoint": "/encrypt", "method": "POST"},
        {"name": "validation_send_api", "port": 8005, "endpoint": "/validation/send", "method": "POST"},
    ]
    
    results = []
    
    import datetime as dt_root
    current_data["flowMetrics"] = {
        "request_id": request_id,
        "requestId": request_id,
        "total_steps": len(pipeline),
        "timestamp": dt_root.datetime.now().isoformat(),
        "pipeline_trace": [{"service": s["name"], "status": "pending", "port": s["port"]} for s in pipeline]
    }

    if isinstance(current_data, dict):
        for k, v in current_data.items():
            if isinstance(v, str) and "10.0.0.5" in v:
                current_data[k] = v.replace("10.0.0.5", "192.168.18.18")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for step in pipeline:
            name = step["name"]
            
            # Professional URL & IP Sanitization
            host_ip = "192.168.18.18"
            if isinstance(current_data, dict):
                for k, v in current_data.items():
                    if isinstance(v, str) and ("10.0.0.5" in v or "localhost" in v or "127.0.0.1" in v):
                        current_data[k] = v.replace("10.0.0.5", host_ip).replace("localhost", host_ip).replace("127.0.0.1", host_ip)
            
            url = f"http://{host_ip}:{step['port']}{step['endpoint']}"

            step_result = {
                "service": name,
                "url": url,
                "status": "in_progress",
                "port": step["port"],
                "start_time": datetime.now().isoformat(),
                "raw_response": None
            }

            # Atomic update for real-time dashboard visibility
            results.append(step_result)
            
            import datetime as dt_at
            # Professional Fix: Always show FULL pipeline in trace, update status of current step
            trace = []
            for s in pipeline:
                status = "pending"
                # If service already processed, get its real status
                past_match = next((r for r in results if r["service"] == s["name"]), None)
                if past_match:
                    status = past_match["status"]
                elif s["name"] == name:
                    status = "in_progress"
                
                trace.append({"service": s["name"], "status": status, "port": s["port"]})

            current_data["flowMetrics"] = {
                "total_steps": len(pipeline),
                "timestamp": dt_at.datetime.now().isoformat(),
                "negotiated_algorithm": current_data.get("selectedAlgorithm") or current_data.get("selected_algorithm"),
                "pipeline_trace": trace,
                "mlMetadata": current_data.get("mlMetadata", {})
            }

            try:
                svc = CONFIG.get("services", {}).get(name, {})
                port = svc.get("port", step["port"])
                
                # [PhD Sincronização de IDs]
                # Garante que todos os payloads, independente da customização, carreguem os IDs vitais
                current_data["request_id"] = request_id
                current_data["requestId"] = request_id
                current_data.setdefault("session_id", current_data.get("sessionId", "pending-session"))
                current_data.setdefault("sessionId", current_data.get("session_id"))

                payload = current_data
                if name == "classification_agent":
                    payload_data = current_data.get("data", {})
                    if isinstance(payload_data, str):
                        try: payload_data = json.loads(payload_data)
                        except: pass
                    payload = {"data": payload_data, "request_id": current_data.get("request_id")}
                    payload["requestId"] = request_id
                
                
                if name == "key_destination_engine":
                    negotiation = current_data.get("negotiation") or {}
                    
                    actual_nonce = current_data.get("nonce_b64") or current_data.get("crypto_nonce_b64") or negotiation.get("crypto_nonce_b64") or negotiation.get("nonce_b64")
                    actual_ciphertext = current_data.get("ciphertext_b64") or current_data.get("crypto_ciphertext_b64") or negotiation.get("crypto_ciphertext_b64") or negotiation.get("ciphertext_b64")
                    actual_algo = current_data.get("algorithm") or negotiation.get("selected_algorithm") or current_data.get("selected_algorithm") or "AES256_GCM"
                    actual_sess = current_data.get("session_id") or negotiation.get("session_id") or "null-session"
                    actual_req = current_data.get("request_id") or current_data.get("requestId") or "req-" + str(int(time.time()))
                    
                    km_material = negotiation.get("key_material") or current_data.get("key_material") or ""
                    km_algo = negotiation.get("selected_algorithm") or actual_algo
                    km_expires = negotiation.get("expires_at") or current_data.get("expires_at") or 0

                    if isinstance(km_expires, str):
                        try:
                            clean_date = km_expires.replace("Z", "").split(".")[0]
                            km_expires = int(datetime.fromisoformat(clean_date).timestamp())
                        except: km_expires = int(time.time() + 3600)
                    else: km_expires = int(km_expires)

                    payload = {
                        "session_id": str(actual_sess),
                        "request_id": str(actual_req),
                        "destination": "http://192.168.18.18:8005/validation/send",
                        "delivery_method": "API",
                        "key_material": str(km_material),
                        "algorithm": str(km_algo),
                        "expires_at": int(km_expires),
                        "metadata": {
                            "body": {
                                "requestId": str(actual_req),
                                "sessionId": str(actual_sess),
                                "selectedAlgorithm": str(km_algo),
                                "cryptoNonceB64": str(actual_nonce),
                                "cryptoCiphertextB64": str(actual_ciphertext),
                                "cryptoAlgorithm": str(actual_algo),
                                "cryptoExpiresAt": int(km_expires),
                                "originUrl": str(current_data.get("originUrl") or "http://192.168.18.18:8090/callback"),
                                "sourceId": str(current_data.get("sourceId") or "agent-wsl-01")
                            },
                            "original_destination": current_data.get("destination"),
                            "risk_label": current_data.get("risk_label") or "Unknown",
                            "pipeline_sync": True
                        }
                    }
                    
                # Injeção final de segurança para garantir agrupamento no Dashboard
                if isinstance(payload, dict):
                    payload["request_id"] = request_id
                    payload["requestId"] = request_id

                if name == "validation_send_api":
                    negotiation = current_data.get("negotiation") or {}
                    
                    actual_nonce = current_data.get("nonce_b64") or current_data.get("crypto_nonce_b64") or negotiation.get("crypto_nonce_b64")
                    actual_ciphertext = current_data.get("ciphertext_b64") or current_data.get("crypto_ciphertext_b64") or negotiation.get("crypto_ciphertext_b64")
                    actual_algo = current_data.get("algorithm") or current_data.get("selected_algorithm") or negotiation.get("selected_algorithm") or current_data.get("selectedAlgorithm")
                    actual_sess = current_data.get("session_id") or negotiation.get("session_id") or current_data.get("sessionId")
                    actual_req = current_data.get("request_id") or current_data.get("requestId") or negotiation.get("request_id")
                    actual_origin = current_data.get("originUrl") or "http://192.168.18.18:8090/callback"
                    actual_expires = current_data.get("expires_at") or negotiation.get("expires_at") or 0

                    if isinstance(actual_expires, str):
                        try:
                            clean_date = actual_expires.replace("Z", "").split(".")[0]
                            actual_expires = int(datetime.fromisoformat(clean_date).timestamp())
                        except: actual_expires = int(time.time() + 3600)

                    payload = {
                        "requestId": str(actual_req),
                        "sessionId": str(actual_sess),
                        "selectedAlgorithm": str(actual_algo) if actual_algo else "AES256_GCM",
                        "cryptoNonceB64": str(actual_nonce) if actual_nonce and actual_nonce != "None" else "AAAAAAAAAAAAAA==",
                        "cryptoCiphertextB64": str(actual_ciphertext) if actual_ciphertext and actual_ciphertext != "None" else "AAAAAAAAAAAAAA==",
                        "cryptoAlgorithm": str(actual_algo),
                        "cryptoExpiresAt": int(actual_expires),
                        "originUrl": str(actual_origin),
                        "sourceId": str(current_data.get("sourceId") or "agent-wsl-01"),
                        "keyMaterial": str(current_data.get("key_material") or negotiation.get("key_material") or "")
                    }

                if name == "risk_service":
                    payload = {
                        "single": {
                            "features": current_data.get("data", {})
                        },
                        "models": ["random_forest", "logistic_regression", "lightgbm"],
                        "version": "v20260107_202018",
                        "include_prob": True
                    }

                if name == "rl_engine":
                    src = current_data.get("source")
                    dst = current_data.get("destination")
                    
                    if isinstance(src, dict):
                        payload["source"] = src.get("ip") or "192.168.18.18"
                    else:
                        payload["source"] = str(src or "192.168.18.18")
                        
                    if isinstance(dst, dict):
                        payload["destination"] = dst.get("ip") or "192.168.18.18"
                    else:
                        payload["destination"] = str(dst or "192.168.18.18")

                    risk_data = current_data.get("risk") or {}
                    conf_data = current_data.get("confidentiality") or {}
                    
                    risk = risk_data.get("score") or current_data.get("risk_score")
                    conf = conf_data.get("score") or current_data.get("conf_score")
                    
                    if not risk and current_data.get("results"):
                        label = current_data["results"][0].get("label", "").lower()
                        if "high" in label: risk = 0.8
                        elif "low" in label: risk = 0.2
                    
                    payload["risk_score"] = float(risk or 0.5)
                    payload["conf_score"] = float(conf or 0.5)
                    
                    security_lvl = current_data.get("security_level") or risk_data.get("level") or "moderate"
                    payload["security_level"] = str(security_lvl).upper()
                    
                    payload["metadata"] = {
                        "risk_label": current_data.get("results", [{}])[0].get("label"),
                        "context_version": current_data.get("version")
                    }
                
                if name == "crypto_module":
                    negotiation = current_data.get("negotiation") or {}
                    sess_id = current_data.get("session_id") or negotiation.get("session_id")
                    req_id = current_data.get("request_id") or negotiation.get("request_id")
                    
                    p_b64 = current_data.get("plaintext_b64")
                    if not p_b64 and "data" in current_data:
                        import base64
                        msg_data = current_data.get("data")
                        msg_str = json.dumps(msg_data) if isinstance(msg_data, (dict, list)) else str(msg_data)
                        p_b64 = base64.b64encode(msg_str.encode()).decode()

                    payload = {
                        "request_id": req_id,
                        "session_id": sess_id,
                        "plaintext_b64": p_b64,
                        "fetch_from_interceptor": False,
                        "key_material": negotiation.get("key_material") or current_data.get("key_material"),
                        "algorithm": negotiation.get("selected_algorithm") or current_data.get("selected_algorithm") or "AES256_GCM",
                        "nonce_b64": negotiation.get("crypto_nonce_b64") or current_data.get("crypto_nonce_b64")
                    }
                
                if name == "risk_service":
                    payload = {
                        "single": {
                            "features": current_data.get("data", {})
                        },
                        "models": ["random_forest", "logistic_regression", "lightgbm"],
                        "version": "v20260107_202018"
                    }
                
                if name == "confiability_service":
                    payload = {
                        "request_id": current_data.get("request_id"),
                        "data": current_data.get("data", {}),
                        "context": current_data.get("source", {}),
                        "classification_level": current_data.get("confidentiality", {}).get("classification", "internal")
                    }

                if name == "kms":
                    payload = {
                        "request_id": current_data.get("request_id"),
                        "session_id": current_data.get("session_id"),
                        "source": payload.get("source", "unknown"),
                        "destination": payload.get("destination", "unknown"),
                        "algorithm": current_data.get("selected_algorithm") or "AES256_GCM",
                        "security_level": current_data.get("security_level") or "moderate"
                    }

                headers = {}
                if name == "classification_agent":
                    api_key = svc.get("env", {}).get("CLASSIFY_API_KEY") or os.environ.get("CLASSIFY_API_KEY", "your-api-key-for-authentication")
                    headers["X-API-Key"] = api_key

                response = await client.request(
                    step["method"],
                    url,
                    json=payload,
                    headers=headers,
                    follow_redirects=True
                )
                
                step_result["status_code"] = response.status_code

                # Update atomic flow metrics for dashboard sync
                import datetime as dt_iter
                full_trace = []
                for s in pipeline:
                    status = "pending"
                    past_match = next((r for r in results if r["service"] == s["name"]), None)
                    if past_match: status = past_match["status"]
                    full_trace.append({"service": s["name"], "status": status, "port": s["port"]})

                current_data["flowMetrics"]["pipeline_trace"] = full_trace

                if response.status_code == 200:
                    step_result["status"] = "success"
                    try:
                        resp_json = response.json()
                        step_result["response"] = resp_json
                        step_result["raw_response"] = resp_json  # PhD Monitoring
                        step_result["end_time"] = datetime.now().isoformat()

                        if isinstance(resp_json, dict):
                            if name == "risk_service":
                                current_data["risk_v2_details"] = resp_json
                            
                            if name == "classification_agent":
                                if "results" in resp_json:
                                    current_data["classification_results"] = resp_json["results"]
                                    if len(resp_json["results"]) > 0:
                                        current_data["risk_label"] = resp_json["results"][0].get("label")
                            
                            if "destination" not in current_data or current_data["destination"] == "server-backend":
                                current_data["destination"] = "http://192.168.18.18:8005/validation/send"
                            
                            resp_json["destination"] = current_data["destination"]

                            if isinstance(current_data.get("destination"), str) and "10.0.0.5" in current_data["destination"]:
                                current_data["destination"] = current_data["destination"].replace("10.0.0.5", "192.168.18.18")
                            if current_data.get("destination") == "192.168.18.18":
                                current_data["destination"] = "http://192.168.18.18:8005/validation/send"

                            if "selected_algorithm" in resp_json and "algorithm" not in resp_json:
                                resp_json["algorithm"] = resp_json["selected_algorithm"]
                            
                            if name == "kms" or name == "kms_service":
                                resp_json["fetch_from_interceptor"] = False
                                current_data["selected_algorithm"] = resp_json.get("selected_algorithm") or resp_json.get("algorithm")
                                current_data["selectedAlgorithm"] = current_data["selected_algorithm"]
                                if "data" in current_data and isinstance(current_data["data"], dict):
                                    import base64
                                    msg_str = json.dumps(current_data["data"])
                                    resp_json["plaintext_b64"] = base64.b64encode(msg_str.encode()).decode()
                                    resp_json["data"] = current_data["data"]
                            
                            if name == "risk_service":
                                current_data["risk_v2_details"] = resp_json
                                current_data["risk_score"] = resp_json.get("risk_score")
                                if "models" in resp_json:
                                    current_data["models"] = resp_json["models"]

                            if name == "classification_agent":
                                try:
                                    class_results = resp_json.get("results") or resp_json.get("classification_results") or []
                                    if class_results:
                                        current_data["classification_results"] = class_results
                                        current_data["model_name"] = resp_json.get("model_name")
                                        current_data["conf_score"] = class_results[0].get("confidence") or class_results[0].get("score", 0.0)
                                        current_data["classification_confidence"] = current_data["conf_score"]
                                except: pass

                            if name == "confiability_service":
                                current_data["confidentiality"] = resp_json
                                if "score" in resp_json:
                                    current_data["conf_score"] = resp_json["score"]

                            if name == "crypto_module":
                                if "ciphertext_b64" in resp_json:
                                    current_data["cryptoCiphertextB64"] = resp_json["ciphertext_b64"]
                                    current_data["ciphertext_b64"] = resp_json["ciphertext_b64"]
                                if "nonce_b64" in resp_json:
                                    current_data["cryptoNonceB64"] = resp_json["nonce_b64"]
                                    current_data["nonce_b64"] = resp_json["nonce_b64"]
                                if "algorithm" in resp_json:
                                    current_data["cryptoAlgorithm"] = resp_json["algorithm"]

                            if name == "handshake_negotiator":
                                h_metrics = {
                                    "selected_model": resp_json.get("model_name") or resp_json.get("selected_model") or "KMS-Handshake-v1",
                                    "pqc_enabled": any(x in str(resp_json.get("selected_algorithm", "")).lower() for x in ["kyber", "frodo", "dilithium", "saber", "bike"]),
                                    "latency_ms": resp_json.get("prediction_time_ms") or resp_json.get("negotiation_time_ms"),
                                    "algorithm": resp_json.get("selected_algorithm") or resp_json.get("algorithm")
                                }
                                current_data["handshake_metrics"] = h_metrics
                                final_alg = h_metrics["algorithm"]
                                current_data["selectedAlgorithm"] = final_alg
                                current_data["selected_algorithm"] = final_alg
                                if h_metrics["algorithm"]:
                                    current_data["selected_algorithm"] = h_metrics["algorithm"]
                                if resp_json.get("session_id"):
                                    current_data["session_id"] = resp_json.get("session_id")
                                    resp_json["sessionId"] = resp_json.get("session_id")
                                if resp_json.get("expires_at"):
                                    current_data["expires_at"] = resp_json.get("expires_at")

                            if name == "rl_engine":
                                rl_final = resp_json.get("payload", {}) if "payload" in resp_json else resp_json
                                rl_metrics = {
                                    "version": resp_json.get("version") or rl_final.get("rl_engine_version") or "2.0",
                                    "decision": rl_final.get("selected_algorithm") or rl_final.get("decision"),
                                    "security_level": rl_final.get("security_level") or resp_json.get("security_level")
                                }
                                current_data["rl_metrics"] = rl_metrics
                                if rl_final.get("security_level"):
                                    current_data["security_level"] = rl_final.get("security_level")

                            if name == "context_api":
                                import datetime as dt_mod
                                final_alg = current_data.get("selected_algorithm") or current_data.get("algorithm") or current_data.get("selectedAlgorithm")
                                
                                risk_details = current_data.get("risk_v2_details") or {}
                                class_results = current_data.get("classification_results", [])
                                rl_info = current_data.get("rl_metrics") or {}
                                hand_info = current_data.get("handshake_metrics") or {}

                                try:
                                    import httpx as httpx_met
                                    print(f"\n[PHD-DEBUG] Sincronizando com 192.168.18.18...")
                                    for svc_name, svc_port, endpoints in [
                                        ("risk_v2", 8003, ["/prediction/metrics", "/metrics/latest", "/metrics"]),
                                        ("classification", 8088, ["/classification/metrics", "/stats", "/metrics"]),
                                        ("rl_engine", 9009, ["/metrics"])
                                    ]:
                                        try:
                                            m_data = None
                                            for svc_endpoint in endpoints:
                                                m_resp = httpx_met.get(f"http://192.168.18.18:{svc_port}{svc_endpoint}", timeout=1.0)
                                                if m_resp.status_code == 200:
                                                    m_data = m_resp.json()
                                                    print(f" -> {svc_name} OK via {svc_endpoint}")
                                                    break
                                            
                                            if m_data:
                                                if svc_name == "risk_v2":
                                                    risk_details["realtime_metrics"] = m_data
                                                    if "models" in m_data: current_data["risk_v2_details"] = m_data
                                                if svc_name == "classification":
                                                    class_results = [m_data]
                                                if svc_name == "rl_engine":
                                                    rl_info["realtime_metrics"] = m_data
                                                    usage = m_data.get("metrics", {}).get("algorithm_usage", {})
                                                    actual_decision = next((k for k, v in usage.items() if v > 0), "AES256_GCM")
                                                    current_data["rl_metrics"] = {
                                                        "version": m_data.get("version", "2.0"),
                                                        "avg_q_value": m_data.get("q_table_stats", {}).get("avg_q_value", 0),
                                                        "decision": actual_decision
                                                    }
                                        except Exception as e:
                                            print(f" -> {svc_name} ({svc_port}): ERRO - {str(e)}")
                                except:
                                    pass
                            
                            ml_metadata = {
                                "risk_v2": current_data.get("models") or risk_details.get("models", {}),
                                "classification": {
                                    "model": current_data.get("model_name", "logreg_lbfgs_v2"),
                                    "confidence": current_data.get("classification_confidence") or current_data.get("conf_score", 0.0),
                                    "results": class_results[0] if class_results else {}
                                },
                                "confiability": {
                                    "model": "conf-fb-0.0.1",
                                    "score": current_data.get("conf_score", 0.4),
                                    "details": current_data.get("confidentiality", {})
                                },
                                "rl_engine": rl_info,
                                "handshake": hand_info,
                                "ia_version": current_data.get("version", "v20260107_202018")
                            }
                            
                            current_data["mlMetadata"] = ml_metadata
                            
                            if "rl_engine" in ml_metadata and "realtime_metrics" in ml_metadata["rl_engine"]:
                                rl_realtime = ml_metadata["rl_engine"]["realtime_metrics"]
                                resp_json["rl_metrics"] = {
                                    "version": rl_realtime.get("version", "2.0"),
                                    "decision": next((k for k, v in rl_realtime.get("metrics", {}).get("algorithm_usage", {}).items() if v > 0), "AES256_GCM"),
                                    "avg_q_value": rl_realtime.get("q_table_stats", {}).get("avg_q_value", 0),
                                    "security_level": current_data.get("security_level", "LOW")
                                }
                            resp_json["originUrl"] = current_data.get("originUrl") or "http://192.168.18.18:8090/callback"
                            
                            current_data.update(resp_json)
                        
                    except Exception as e:
                        pass
                else:
                    step_result["status"] = "error"
                    step_result["error"] = response.text[:500]
                    break
                    
            except Exception as e:
                step_result["status"] = "error"
                step_result["error"] = str(e)
                break
                
    import datetime as dt_final
    trace = []
    completed_services = {r["service"]: r["status"] for r in results}
    for s in pipeline:
        status = completed_services.get(s["name"], "pending")
        trace.append({"service": s["name"], "status": status, "port": s["port"]})

    current_data["flowMetrics"] = {
        "total_steps": len(pipeline),
        "timestamp": dt_final.datetime.now().isoformat(),
        "negotiated_algorithm": current_data.get("selectedAlgorithm") or current_data.get("selected_algorithm"),
        "pipeline_trace": trace
    }

    current_data["mlMetadata"] = {
        "risk_v2": current_data.get("models") or [],
        "classification": {
            "model": current_data.get("model_name", "logreg_lbfgs_v2"),
            "confidence": current_data.get("classification_confidence") or current_data.get("conf_score", 0.0)
        },
        "rl_engine": current_data.get("rl_metrics") or {},
        "handshake": current_data.get("handshake_metrics") or {},
        "ia_version": current_data.get("version", "v20260107_202018")
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "pipeline_results": results,
        "final_data": current_data
    }

if __name__ == "__main__":
    uvicorn.run("orchestrator_linux:APP", host="0.0.0.0", port=8090, reload=False)