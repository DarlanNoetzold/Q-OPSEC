import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    

    def __init__(self, base_dir: str | Path = "config") -> None:
        self.base_dir = Path(base_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(self, filename: str) -> Dict[str, Any]:
        
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
        
        if len(self._cache) == 1:
            return list(self._cache.values())[0]

        return self._cache if self._cache else {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        
        parts = key.split(".")
        if not parts:
            return default

        root_name = parts[0]
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