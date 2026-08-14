from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    api_title: str = "Classification Agent API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8088

    secret_key: str = "your-super-secret-key-change-this-in-production"
    api_key: str = "your-api-key-for-authentication"
    access_token_expire_minutes: int = 30

    mongodb_url: str = "mongodb://daily:daily123@localhost:27017/daily?authSource=admin "
    mongodb_database: str = "classification_agent"

    ml_registry_dir: str = "/mnt/c/Projetos/Q-OPSEC/classify_scheduler/model_registry"
    ml_registry_latest_file: str = "latest.json"
    auto_reload_ml: bool = True


    enable_metrics: bool = True
    metrics_port: int = 9090

    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    log_level: str = "INFO"
    log_file: Optional[str] = None
    is_development: bool = True

    allowed_hosts: list = ["*"]

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ()
    }


settings = Settings()