import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any, Optional
from service import ImprovedRLEngineService
from negotiator_client import send_to_handshake
import traceback


class Settings:
    host: str = "localhost"
    port: int = 9009
    debug: bool = False
    log_level: str = "info"
    registry_path: str = "./rl_registry.json"
    handshake_url: str = "http://localhost:8001/handshake"

    use_dqn: bool = False
    policy_type: str = "context_aware"
    training_mode: bool = True


settings = Settings()

app = FastAPI(
    title="Enhanced RL Engine Service",
    description="Adaptive cryptographic algorithm selection using Reinforcement Learning",
    version="2.0.0"
)

rl_service = ImprovedRLEngineService(
    registry_path=Path(settings.registry_path),
    use_dqn=settings.use_dqn,
    policy_type=settings.policy_type
)
rl_service.set_training_mode(settings.training_mode)


class ContextRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    source: str = Field(..., description="Source node identifier")
    destination: str = Field(..., description="Destination node identifier")
    security_level: str = Field(..., description="Security level classification")
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Risk score (0-1)")
    conf_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidentiality score (0-1)")
    dst_props: Dict[str, Any] = Field(default_factory=dict, description="Destination properties")

    # Optional enhanced fields
    source_reputation: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_location_risk: Optional[float] = Field(None, ge=0.0, le=1.0)
    data_sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    service_criticality: Optional[float] = Field(None, ge=0.0, le=1.0)
    time_of_day: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    is_peak_attack_time: Optional[bool] = None
    current_threat_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    system_load: Optional[float] = Field(None, ge=0.0, le=1.0)
    available_resources: Optional[float] = Field(None, ge=0.0, le=1.0)
    network_latency: Optional[float] = Field(None, ge=0.0)


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="Request identifier")
    success: bool = Field(..., description="Whether negotiation succeeded")
    latency: Optional[float] = Field(None, description="Latency in milliseconds")
    resource_usage: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_message: Optional[str] = None


@app.post("/act",
          summary="Select cryptographic algorithm",
          description="Main endpoint - selects optimal algorithm and initiates handshake")
def act(req: ContextRequest):
    try:
        print(f"\\n{'=' * 60}")
        print(f"DEBUG: Received request")
        print(f"  request_id: {req.request_id}")
        print(f"  risk_score: {req.risk_score} (type: {type(req.risk_score)})")
        print(f"  conf_score: {req.conf_score} (type: {type(req.conf_score)})")
        print(f"{'=' * 60}\\n")

        req_dict = req.model_dump()
        print(f"DEBUG: req.model_dump() = {req_dict}")
        print(f"  risk_score in dict: {req_dict.get('risk_score')} (type: {type(req_dict.get('risk_score'))})")
        print(f"  conf_score in dict: {req_dict.get('conf_score')} (type: {type(req_dict.get('conf_score'))})")

        payload = rl_service.build_negotiation_payload(req_dict)

        result = send_to_handshake(settings.handshake_url, payload)

        if "success" in result:
            outcome = {
                "success": result.get("success", False),
                "latency": result.get("latency", 0.0),
                "resource_usage": result.get("resource_usage", 0.5)
            }
            rl_service.process_feedback(req.request_id, outcome)

        return {
            "request_id": req.request_id,
            "negotiation": result,
            "payload": payload,
            "status": "success"
        }

    except Exception as e:
        print(f"\\n{'!' * 60}")
        print(f"ERROR: Exception occurred")
        print(f"  Error message: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        print(f"\\nFull traceback:")
        traceback.print_exc()
        print(f"{'!' * 60}\\n")

        # Process failure feedback
        outcome = {
            "success": False,
            "latency": 0.0,
            "resource_usage": 0.0
        }
        try:
            rl_service.process_feedback(req.request_id, outcome)
        except:
            pass

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback",
          summary="Provide feedback",
          description="Provide feedback on negotiation outcome for learning")
def feedback(req: FeedbackRequest):
    try:
        outcome = {
            "success": req.success,
            "latency": req.latency or 0.0,
            "resource_usage": req.resource_usage or 0.5
        }

        rl_service.process_feedback(req.request_id, outcome)

        return {
            "request_id": req.request_id,
            "status": "feedback_processed"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/episode/end",
          summary="End training episode",
          description="Mark end of training episode and save learned policy")
def end_episode():
    try:
        rl_service.end_episode()

        return {
            "status": "episode_ended",
            "episode_count": rl_service.episode_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics",
         summary="Get performance metrics",
         description="Retrieve current performance metrics")
def get_metrics():
    try:
        metrics = rl_service.get_metrics()
        q_stats = rl_service.get_q_table_stats()

        return {
            "metrics": metrics,
            "q_table_stats": q_stats,
            "episode_count": rl_service.episode_count,
            "training_mode": rl_service.training_mode
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/enable",
          summary="Enable training mode",
          description="Enable training mode for learning")
def enable_training():
    rl_service.set_training_mode(True)
    return {"status": "training_enabled"}


@app.post("/training/disable",
          summary="Disable training mode",
          description="Disable training mode (inference only)")
def disable_training():
    rl_service.set_training_mode(False)
    return {"status": "training_disabled"}


@app.get("/policy/export",
         summary="Export learned policy",
         description="Export learned policy to file")
def export_policy(path: str = "./exported_policy.json"):
    try:
        rl_service.export_policy(Path(path))
        return {
            "status": "policy_exported",
            "path": path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/policy/import",
          summary="Import learned policy",
          description="Import learned policy from file")
def import_policy(path: str = "./exported_policy.json"):
    try:
        rl_service.import_policy(Path(path))
        return {
            "status": "policy_imported",
            "path": path
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Policy file not found: {path}")


@app.get("/health",
         summary="Health check",
         description="Check service health")
def health_check():
    return {
        "status": "healthy",
        "service": "RL Engine",
        "version": "2.0.0",
        "training_mode": rl_service.training_mode,
        "policy_type": rl_service.policy_type
    }


@app.get("/",
         summary="Service info",
         description="Get service information")
def root():
    return {
        "service": "Enhanced RL Engine",
        "version": "2.0.0",
        "description": "Adaptive cryptographic algorithm selection using Reinforcement Learning",
        "endpoints": {
            "POST /act": "Select algorithm and initiate handshake",
            "POST /feedback": "Provide feedback for learning",
            "POST /episode/end": "End training episode",
            "GET /metrics": "Get performance metrics",
            "POST /training/enable": "Enable training mode",
            "POST /training/disable": "Disable training mode",
            "GET /policy/export": "Export learned policy",
            "POST /policy/import": "Import learned policy",
            "GET /health": "Health check"
        },
        "features": [
            "Q-Learning and Deep Q-Network (DQN) support",
            "Multiple exploration strategies (epsilon-greedy, Boltzmann, UCB, etc.)",
            "Context-aware algorithm selection",
            "Quantum and post-quantum cryptography support",
            "Online learning with experience replay",
            "Policy export/import for transfer learning",
            "Comprehensive performance metrics"
        ]
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced RL Engine Service v2.0")
    print("=" * 60)
    print(f"Policy Type: {settings.policy_type}")
    print(f"Using DQN: {settings.use_dqn}")
    print(f"Training Mode: {settings.training_mode}")
    print(f"Registry Path: {settings.registry_path}")
    print(f"Handshake URL: {settings.handshake_url}")
    print("=" * 60)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower()
    )