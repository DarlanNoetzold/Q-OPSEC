from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Classification Agent API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080

    # Security
    secret_key: str = "your-super-secret-key-change-this-in-production"
    api_key: str = "your-api-key-for-authentication"
    access_token_expire_minutes: int = 30

    # MongoDB Configuration
    mongodb_url: str = "mongodb://daily:daily123@localhost:27017/daily?authSource=admin "
    mongodb_database: str = "classification_agent"

    # ML Model Registry - renomeado (evita conflitos "model_")
    ml_registry_dir: str = r"C:\Projetos\Q-OPSEC\classify_scheduler\model_registry"
    ml_registry_latest_file: str = "latest.json"
    auto_reload_ml: bool = True
    ml_cache_ttl: int = 300

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    is_development: bool = True

    # CORS
    allowed_hosts: list = ["*"]

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ()  # evita conflito com "model_"
    }


settings = Settings()