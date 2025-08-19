# repositories/config_repo.py
import json
import os
from typing import Dict, Any, Optional

MODELS_DIR = "models"
DATA_DIR = "data"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


def read_registry() -> Dict[str, Any]:
    """Lê o registry de modelos do disco"""
    if not os.path.exists(REGISTRY_PATH):
        return {
            "models": [],  # modelos do risk service
            "best_model": None,  # melhor modelo risk
            "conf_models": [],  # modelos do confidentiality service
            "best_conf_model": None  # melhor modelo confidentiality
        }
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "models": [],
            "best_model": None,
            "conf_models": [],
            "best_conf_model": None
        }


def write_registry(registry: Dict[str, Any]) -> None:
    """Escreve o registry de modelos no disco"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write registry: {e}")


def set_best_model(model_info: Dict[str, Any], service_type: str = "risk") -> None:
    """
    Marca um modelo como o melhor para um serviço específico

    Args:
        model_info: Informações do modelo (name, path, metrics, etc.)
        service_type: "risk" ou "confidentiality"
    """
    registry = read_registry()

    if service_type == "risk":
        # Remove flag best anterior dos modelos de risco
        for m in registry.get("models", []):
            m.pop("best", None)

        # Marca o novo melhor
        model_info["best"] = True

        # Atualiza/insere na lista
        updated = [m for m in registry.get("models", []) if m.get("path") != model_info["path"]]
        updated.append(model_info)
        registry["models"] = sorted(updated, key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)
        registry["best_model"] = model_info

    elif service_type == "confidentiality":
        # Remove flag best anterior dos modelos de confidencialidade
        for m in registry.get("conf_models", []):
            m.pop("best", None)

        # Marca o novo melhor
        model_info["best"] = True

        # Atualiza/insere na lista
        updated = [m for m in registry.get("conf_models", []) if m.get("path") != model_info["path"]]
        updated.append(model_info)
        registry["conf_models"] = sorted(updated, key=lambda x: x.get("metrics", {}).get("f1_macro", 0), reverse=True)
        registry["best_conf_model"] = model_info

    write_registry(registry)


def get_best_model_info(service_type: str = "risk") -> Optional[Dict[str, Any]]:
    """
    Retorna informações do melhor modelo para um serviço específico

    Args:
        service_type: "risk" ou "confidentiality"

    Returns:
        Dict com informações do modelo ou None se não existir
    """
    registry = read_registry()

    if service_type == "risk":
        return registry.get("best_model")
    elif service_type == "confidentiality":
        return registry.get("best_conf_model")

    return None


def cleanup_old_models(max_models_per_service: int = 10) -> None:
    """
    Remove modelos antigos do disco e registry para economizar espaço

    Args:
        max_models_per_service: Número máximo de modelos a manter por serviço
    """
    registry = read_registry()

    # Cleanup modelos de risco
    risk_models = registry.get("models", [])
    if len(risk_models) > max_models_per_service:
        # Mantém os melhores modelos
        risk_models_sorted = sorted(risk_models, key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)
        models_to_remove = risk_models_sorted[max_models_per_service:]

        for model in models_to_remove:
            model_path = model.get("path")
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"Removed old risk model: {model_path}")
                except Exception as e:
                    print(f"Failed to remove {model_path}: {e}")

        registry["models"] = risk_models_sorted[:max_models_per_service]

    # Cleanup modelos de confidencialidade
    conf_models = registry.get("conf_models", [])
    if len(conf_models) > max_models_per_service:
        # Mantém os melhores modelos
        conf_models_sorted = sorted(conf_models, key=lambda x: x.get("metrics", {}).get("f1_macro", 0), reverse=True)
        models_to_remove = conf_models_sorted[max_models_per_service:]

        for model in models_to_remove:
            model_path = model.get("path")
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"Removed old confidentiality model: {model_path}")
                except Exception as e:
                    print(f"Failed to remove {model_path}: {e}")

        registry["conf_models"] = conf_models_sorted[:max_models_per_service]

    write_registry(registry)


class ConfigRepository:
    """Configurações para os serviços de risco e confidencialidade"""

    def get_policy_thresholds(self):
        """Thresholds para classificação de níveis de risco"""
        return {
            "very_low": 0.15,
            "low": 0.35,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        }

    def get_policy_overrides(self, level: str):
        """Políticas de segurança baseadas no nível de risco"""
        table = {
            "very_low": [],
            "low": [],
            "medium": ["enforce_mtls"],
            "high": ["enforce_mtls", "rotate_keys_24h", "pqc_required"],
            "critical": ["enforce_mtls", "rotate_keys_6h", "pqc_required", "block_high_risk_ops"]
        }
        return table.get(level, [])

    def get_confidentiality_thresholds(self):
        """Thresholds para classificação de confidencialidade"""
        return {
            "public": 0.3,
            "internal": 0.5,
            "confidential": 0.7,
            "restricted": 0.9
        }

    def get_dlp_patterns_config(self):
        """Configuração para padrões DLP"""
        return {
            "credit_card": {
                "enabled": True,
                "confidence_boost": 0.15,
                "escalate_to": "restricted"
            },
            "email": {
                "enabled": True,
                "confidence_boost": 0.08,
                "escalate_to": "confidential"
            },
            "national_id": {
                "enabled": True,
                "confidence_boost": 0.08,
                "escalate_to": "confidential"
            },
            "iban": {
                "enabled": True,
                "confidence_boost": 0.15,
                "escalate_to": "restricted"
            }
        }

    def get_model_training_config(self):
        """Configuração para treinamento de modelos"""
        return {
            "risk": {
                "default_n_samples": 400,
                "max_samples": 5000,
                "min_samples": 50,
                "retrain_interval_hours": 1,
                "selection_criterion": "accuracy"
            },
            "confidentiality": {
                "default_n_per_class": 100,
                "max_per_class": 500,
                "min_per_class": 20,
                "retrain_interval_hours": 1,
                "selection_criterion": "f1_macro"
            }
        }

    def get_scheduler_config(self):
        """Configuração do scheduler para retreinamento automático"""
        return {
            "risk_retrain": {
                "interval_hours": 1,
                "max_instances": 1,
                "coalesce": True
            },
            "confidentiality_retrain": {
                "interval_hours": 1,
                "max_instances": 1,
                "coalesce": True,
                "start_offset_minutes": 30  # offset para não sobrecarregar
            },
            "cleanup": {
                "interval_hours": 24,
                "max_models_per_service": 10
            }
        }