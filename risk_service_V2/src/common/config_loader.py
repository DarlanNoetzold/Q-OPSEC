"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}


    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file by name.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Dictionary with configuration
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config


    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files.

        Returns:
            Dictionary mapping config names to their contents
        """
        config_files = [
            "dataset_config",
            "user_profiles",
            "fraud_scenarios",
            "llm_config"
        ]

        return {name: self.load(name) for name in config_files}


    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Get a specific configuration value using dot notation.

        Args:
            config_name: Name of config file
            key_path: Dot-separated path to value (e.g., "dataset.num_users")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        config = self.load(config_name)

        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value