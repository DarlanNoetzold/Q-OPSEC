"""
Configuration management for the Classification Agent API.
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings."""

    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_prefix: str = "/api/v1"
    api_title: str = "Classification Agent API"
    api_version: str = "1.0.0"
    api_description: str = "API for ML model classification predictions"

    # Security
    secret_key: str = "change-this-in-production"
    api_key: Optional[str] = None
    access_token_expire_minutes: int = 30
    allowed_hosts: List[str] = ["*"]

    # Model Configuration
    model_registry_dir: str = "./model_registry"
    auto_reload_model: bool = True
    model_cache_ttl: int = 300  # seconds
    max_prediction_batch_size: int = 1000

    # Database
    database_url: str = "sqlite:///./classification_agent.db"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Logging
    log_format: str = "json"
    log_file: Optional[str] = None

    @validator("model_registry_dir")
    def validate_registry_dir(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model registry directory does not exist: {v}")
        return str(path.resolve())

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()