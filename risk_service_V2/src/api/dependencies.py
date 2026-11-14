import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from common.ttl_cache import file_cache_key
from src.api.models.model_manager import ModelManager
from src.common.logger import logger

MAX_CACHED_VERSIONS = int(os.getenv("RISK_MAX_CACHED_VERSIONS", "3"))
SCHEMA_PATH = Path(os.getenv("RISK_SCHEMA_PATH", "config/schema_fields.yaml"))

_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()
_versions: "OrderedDict[tuple, ModelManager]" = OrderedDict()
_versions_lock = threading.Lock()
_schema_cache: Dict[str, Any] = {"key": None, "value": {}}


def get_manager() -> ModelManager:
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is None:
            manager = ModelManager()
            try:
                manager.load()
            except Exception:
                logger.exception("Failed to load models")
                raise
            logger.info(f"ModelManager loaded version: {manager.loaded_version}")
            _manager = manager
    return _manager


def get_versioned_manager(base: ModelManager, version: str) -> ModelManager:
    if base.loaded_version == version:
        return base
    key = (str(base.models_root), str(base.eval_root), version)
    with _versions_lock:
        cached = _versions.get(key)
        if cached is not None:
            _versions.move_to_end(key)
            return cached
        manager = ModelManager(models_root=str(base.models_root), eval_root=str(base.eval_root))
        manager.load(version)
        _versions[key] = manager
        while len(_versions) > MAX_CACHED_VERSIONS:
            _versions.popitem(last=False)
        return manager


def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        return {}
    key = file_cache_key(SCHEMA_PATH)
    if _schema_cache["key"] != key:
        _schema_cache["value"] = yaml.safe_load(SCHEMA_PATH.read_text()) or {}
        _schema_cache["key"] = key
    return _schema_cache["value"]
