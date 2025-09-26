from dataclasses import dataclass
from pathlib import Path

# Ordem de severidade (0 = mais crítico, 5 = menos crítico)
SECURITY_ORDER = ["Critical", "Very High", "High", "Medium", "Low", "Very Low"]

# Mapeamento canônico para normalização
CANONICAL = {
    "very low": "Very Low",
    "verylow": "Very Low",
    "very_low": "Very Low",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "very high": "Very High",
    "veryhigh": "Very High",
    "very_high": "Very High",
    "critical": "Critical",
}

ALLOWED_CLASSES = SECURITY_ORDER  # Usar a ordem de severidade

@dataclass
class DefaultConfig:
    data_path: Path
    registry_dir: Path
    seed: int = 42
    test_size: float = 0.2
    cv_splits: int = 5
    max_models: int = 20
    target_col: str = "security_level_label"
    allowed_classes = ALLOWED_CLASSES