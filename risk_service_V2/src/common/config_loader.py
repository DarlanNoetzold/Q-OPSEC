import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Simple YAML configuration loader with dot-path access.

    Example:
        loader = ConfigLoader("config")
        dataset_cfg = loader.load("dataset_config.yaml")
        num_users = loader.get("dataset_config.generation.num_users")
    """

    def __init__(self, base_dir: str | Path = "config") -> None:
        self.base_dir = Path(base_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(self, filename: str) -> Dict[str, Any]:
        """Load a YAML config file and cache it by stem name.

        Access key will be the filename without extension, e.g.
        "dataset_config.yaml" -> "dataset_config".
        """
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        stem = path.stem
        if stem in self._cache:
            return self._cache[stem]

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[stem] = data
        return data

    def get_all(self) -> dict:
        """Return the full configuration dictionary from cache.

        If cache is empty, returns empty dict. Use load() first to populate.
        """
        # Se o cache tiver apenas 1 arquivo carregado, retorna ele
        if len(self._cache) == 1:
            return list(self._cache.values())[0]

        # Se tiver múltiplos ou nenhum, retorna o cache inteiro
        return self._cache if self._cache else {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a value using dot-separated path.

        Example:
            get("dataset_config.generation.num_users")
        """
        parts = key.split(".")
        if not parts:
            return default

        root_name = parts[0]
        # try to ensure config is loaded
        filename_guess = f"{root_name}.yaml"
        path = self.base_dir / filename_guess
        if path.exists() and root_name not in self._cache:
            self.load(filename_guess)

        current: Any = self._cache.get(root_name)
        if current is None:
            return default

        for part in parts[1:]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current


default_config_loader = ConfigLoader()