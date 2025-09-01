# repositories/config_repo.py
import json
import os
from typing import Dict, Any, Optional

MODELS_DIR = "models"
DATA_DIR = "data"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

def read_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"models": [], "best_model": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_registry(registry: Dict[str, Any]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

def set_best_model(model_info: Dict[str, Any]) -> None:
    registry = read_registry()
    for m in registry.get("models", []):
        m.pop("best", None)
    model_info["best"] = True
    updated = [m for m in registry.get("models", []) if m.get("path") != model_info["path"]]
    updated.append(model_info)
    registry["models"] = sorted(updated, key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)
    registry["best_model"] = model_info
    write_registry(registry)

def get_best_model_info() -> Optional[Dict[str, Any]]:
    reg = read_registry()
    return reg.get("best_model")

class ConfigRepository:
    def get_policy_thresholds(self):
        return {
            "very_low": 0.15,
            "low": 0.35,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        }

    def get_policy_overrides(self, level: str):
        table = {
            "very_low": [],
            "low": [],
            "medium": ["enforce_mtls"],
            "high": ["enforce_mtls", "rotate_keys_24h", "pqc_required"],
            "critical": ["enforce_mtls", "rotate_keys_6h", "pqc_required", "block_high_risk_ops"]
        }
        return table.get(level, [])