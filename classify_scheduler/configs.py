from dataclasses import dataclass
from pathlib import Path

ALLOWED_CLASSES = ["Very Low", "Low", "Medium", "High", "Very High", "Critical"]

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