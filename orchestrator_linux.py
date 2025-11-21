"""
Q-OPSEC Orchestrator - Linux Version with Docker Support
Manages microservices lifecycle (processes + containers), health checks, logs and metrics
"""
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Body, Query
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

# ===== DOCKER UTILITIES =====

def docker_available() -> bool:
    """Check if Docker is available"""
    return shutil.which("docker") is not None

def docker_running() -> bool:
    """Check if Docker daemon is running"""
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
    """Check if container exists"""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            stderr=subprocess.DEVNULL
        )
        return name in out.decode().strip()
    except subprocess.CalledProcessError:
        return False

def docker_container_running(name: str) -> bool:
    """Check if container is running"""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            stderr=subprocess.DEVNULL
        )
        return name in out.decode().strip()
    except subprocess.CalledProcessError:
        return False

def docker_get_container_id(name: str) -> Optional[str]:
    """Get container ID by name"""
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
    """Check if Docker image exists locally"""
    try:
        out = subprocess.check_output(
            ["docker", "images", "-q", image],
            stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except subprocess.CalledProcessError:
        return False

async def docker_pull_image(image: str) -> Dict[str, Any]:
    """Pull Docker image"""
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
    """Start Docker container"""
    name = cfg.get("container_name", cfg.get("name"))
    image = cfg.get("image")

    if not image:
        return {"status": "error", "error": "No image specified"}

    # Check if container already exists
    if docker_container_exists(name):
        if docker_container_running(name):
            container_id = docker_get_container_id(name)
            return {"status": "running", "container_id": container_id, "container_name": name}
        else:
            # Start existing container
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

    # Check if image exists, pull if not
    if not docker_image_exists(image):
        pull_result = await docker_pull_image(image)
        if pull_result["status"] != "pulled":
            return pull_result

    # Build docker run command
    cmd = ["docker", "run"]

    # Add name
    cmd.extend(["--name", name])

    # Add hostname
    if cfg.get("hostname"):
        cmd.extend(["--hostname", cfg["hostname"]])

    # Add environment variables
    for key, value in cfg.get("env", {}).items():
        cmd.extend(["--env", f"{key}={value}"])

    # Add volumes
    for volume in cfg.get("volumes", []):
        cmd.extend(["--volume", volume])

    # Add ports
    for port in cfg.get("ports", []):
        cmd.extend(["-p", port])

    # Add network
    if cfg.get("network"):
        cmd.extend(["--network", cfg["network"]])

    # Add workdir
    if cfg.get("workdir"):
        cmd.extend(["--workdir", cfg["workdir"]])

    # Add restart policy
    restart = cfg.get("restart", "no")
    cmd.extend(["--restart", restart])

    # Add runtime
    if cfg.get("runtime"):
        cmd.extend(["--runtime", cfg["runtime"]])

    # Add labels
    for label in cfg.get("labels", []):
        cmd.extend(["--label", label])

    # Add extra args
    for arg in cfg.get("extra_args", []):
        cmd.append(arg)

    # Detached mode
    cmd.append("-d")

    # Add image
    cmd.append(image)

    # Add command
    if cfg.get("command"):
        if isinstance(cfg["command"], list):
            cmd.extend(cfg["command"])
        else:
            cmd.append(cfg["command"])

    # Execute
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
    """Stop Docker container"""
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
    """Remove Docker container"""
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
    """Get container logs"""
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
    """Get container stats"""
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

# ===== ORIGINAL UTILITIES =====

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
    """Check if PID is running (Linux)"""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def find_pid_by_port(port: int) -> Optional[int]:
    """Find PID listening on port (Linux)"""
    # Try lsof first
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

    # Try fuser as fallback
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

    # Try ss as last resort
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL
        )
        lines = out.decode(errors="ignore").splitlines()
        for line in lines[1:]:  # Skip header
            if f":{port}" in line and "pid=" in line:
                # Extract pid from format: users:(("python",pid=12345,fd=3))
                import re
                match = re.search(r'pid=(\d+)', line)
                if match:
                    return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None

def tail_log(path: str, lines: int = 200) -> List[str]:
    """Get last N lines from log file"""
    p = Path(path)
    if not p.exists():
        return []

    try:
        # Use tail command if available (faster)
        out = subprocess.check_output(
            ["tail", f"-n{lines}", str(p)],
            stderr=subprocess.DEVNULL
        )
        return out.decode(errors="replace").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to Python implementation
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
    """Start a service (process or container)"""
    cfg = service_cfg(name)
    service_type = cfg.get("type", "process")

    # Handle Docker containers
    if service_type == "docker":
        if not docker_running():
            return {"status": "error", "error": "Docker daemon not running"}
        return await docker_start_container(cfg)

    # Handle regular processes
    # Check if already running
    if name in STATE and STATE[name].get("proc"):
        proc = STATE[name]["proc"]
        if hasattr(proc, "returncode") and proc.returncode is None:
            return {"status": "running", "pid": proc.pid}

    # Check pidfile
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

    # Prepare environment
    cwd = (BASE_DIR / cfg["cwd"]).resolve() if cfg.get("cwd") else BASE_DIR
    cmd = cfg["start"]
    logs_dir = Path(CONFIG["paths"]["logs_dir"]).resolve()
    log_file = (BASE_DIR / (cfg.get("log_file") or (logs_dir / f"{name}.log"))).as_posix()

    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    # Open log file
    stdout = open(log_file, "ab", buffering=0)
    stderr = stdout
    env = env_for_service(cfg)

    # Start process
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True  # Detach from parent session
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
    """Try graceful shutdown via actuator endpoint"""
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
    """Stop a service (process or container)"""
    cfg = service_cfg(name)
    service_type = cfg.get("type", "process")

    # Handle Docker containers
    if service_type == "docker":
        container_name = cfg.get("container_name", name)
        return await docker_stop_container(container_name, int(timeout))

    # Handle regular processes
    state = STATE.get(name, {})
    proc = state.get("proc")
    pid = state.get("pid")
    port = cfg.get("port")

    # Try graceful shutdown
    await graceful_shutdown_if_possible(cfg)

    # Get PID from various sources
    if not pid:
        pid = read_pidfile(name)

    if not pid and port:
        pid = find_pid_by_port(int(port))

    if not pid:
        return {"status": "not_running"}

    # Try to terminate process
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass

        # Wait for termination
        start_t = time.time()
        while (time.time() - start_t) < timeout:
            if proc.returncode is not None:
                break
            await asyncio.sleep(0.2)

        # Force kill if still running
        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass

    # Kill by PID if still running
    if pid and is_pid_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)

            # Wait for termination
            start_t = time.time()
            while (time.time() - start_t) < timeout and is_pid_running(pid):
                await asyncio.sleep(0.2)

            # Force kill if still running
            if is_pid_running(pid):
                os.kill(pid, signal.SIGKILL)
                await asyncio.sleep(0.5)
        except Exception:
            pass

    # Kill by port if still occupied
    if port:
        end_time = time.time() + timeout
        while time.time() < end_time:
            port_pid = find_pid_by_port(int(port))
            if port_pid is None:
                break

            if port_pid != pid:
                # Different process on port, kill it
                try:
                    os.kill(port_pid, signal.SIGTERM)
                    await asyncio.sleep(0.5)
                    if find_pid_by_port(int(port)) is not None:
                        os.kill(port_pid, signal.SIGKILL)
                except Exception:
                    pass

            await asyncio.sleep(0.5)

    # Clean up pidfile
    try:
        pf = pidfile_path(name)
        if not pid or not is_pid_running(pid):
            pf.unlink(missing_ok=True)
    except Exception:
        pass

    # Check final status
    still_running = bool(pid and is_pid_running(pid))
    still_on_port = bool(port and find_pid_by_port(int(port)) is not None)

    if not still_running and not still_on_port:
        STATE.pop(name, None)
        return {"status": "stopped", "pid": pid}
    if not still_running and still_on_port:
        return {"status": "killed", "pid": pid}
    return {"status": "failed_to_stop", "pid": pid}

async def restart_service(name: str) -> Dict[str, Any]:
    """Restart a service"""
    await stop_service(name)
    await asyncio.sleep(1.0)
    return await start_service(name)

async def check_health(url: str, timeout: float = 6.0) -> Tuple[bool, Optional[int], Optional[str]]:
    """Check service health endpoint"""
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
    """Load config and restore running services"""
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

# ==== API ENDPOINTS ====

@APP.get("/docker/status")
def docker_status():
    """Check Docker availability"""
    return {
        "available": docker_available(),
        "running": docker_running()
    }

@APP.get("/services")
def list_services():
    """List all services and their status"""
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
            })
    return out

@APP.post("/services/{name}/start")
async def api_start(name: str):
    """Start a service"""
    return await start_service(name)

@APP.post("/services/{name}/stop")
async def api_stop(name: str):
    """Stop a service"""
    return await stop_service(name)

@APP.post("/services/{name}/restart")
async def api_restart(name: str):
    """Restart a service"""
    return await restart_service(name)

@APP.get("/services/{name}/status")
async def api_status(name: str):
    """Get service status with health check"""
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
    """Get service logs"""
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
    """Remove Docker container"""
    cfg = service_cfg(name)
    if cfg.get("type") != "docker":
        raise HTTPException(status_code=400, detail="Service is not a Docker container")

    container_name = cfg.get("container_name", name)
    return await docker_remove_container(container_name, force)

class RequestSpec(BaseModel):
    method: str = Field(..., pattern="(?i)^(GET|POST|PUT|DELETE|PATCH)$")
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    json: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    timeout: float = 15.0

@APP.post("/request")
async def api_request(spec: RequestSpec):
    """Proxy HTTP request to service"""
    try:
        async with httpx.AsyncClient(timeout=spec.timeout) as client:
            r = await client.request(
                spec.method.upper(),
                spec.url,
                headers=spec.headers,
                json=spec.json,
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
    """Start all services in order"""
    order = CONFIG.get("startup_order", list(CONFIG.get("services", {}).keys()))
    results = []
    for name in order:
        results.append({name: await start_service(name)})
        await asyncio.sleep(1.0)
    return results

@APP.post("/stop/all")
async def stop_all():
    """Stop all services in reverse order"""
    names = list(CONFIG.get("services", {}).keys())
    results = []
    for name in reversed(names):
        results.append({name: await stop_service(name)})
    return results

@APP.post("/demo/predict")
async def demo_predict(api_key: Optional[str] = None):
    """Demo prediction request"""
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

# ==== TRACE & FLOW ENDPOINTS ====

import re
from datetime import datetime
from collections import defaultdict

def search_in_log(log_path: str, request_id: str, context_lines: int = 2) -> List[Dict[str, Any]]:
    """Search request_id in log file"""
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
    """Extract timestamp from log line"""
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
    """Trace request_id across all service logs"""
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
    """Show request flow status through pipeline"""
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

    # Determine current step
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
    """Build chronological timeline of request"""
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
    """Find active request IDs in recent logs"""
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

# ==== CONFIG & METRICS ENDPOINTS ====

class ServiceConfigUpdate(BaseModel):
    start: Optional[List[str]] = None
    cwd: Optional[str] = None

@APP.put("/services/{name}/config")
async def update_service_config(name: str, config: ServiceConfigUpdate = Body(...)):
    """Update service configuration"""
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
    """Extract API key from service config"""
    if "env" in svc:
        return svc["env"].get("CLASSIFY_API_KEY") or svc["env"].get("API_KEY")
    return None

async def proxy_get(url, headers=None):
    """Proxy GET request"""
    headers = headers or {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

@APP.get("/metrics/{service_name}/sessions")
async def get_metrics_sessions(service_name: str):
    """Get training/metrics sessions"""
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
    """Get session detail"""
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
    """Get list of metric images"""
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
    """Get metric image"""
    svc = service_cfg(service_name)
    base_url = svc.get("base_url")
    if not base_url:
        raise HTTPException(status_code=400, detail="Service base_url not configured")

    # Check local cache first
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
    """Get available datasets"""
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
    """Preview dataset"""
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

if __name__ == "__main__":
    uvicorn.run("orchestrator_linux:APP", host="0.0.0.0", port=8090, reload=False)