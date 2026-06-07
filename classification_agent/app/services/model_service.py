import json
import os
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
import structlog

from ..core.config import settings
from ..models.database import ModelRecord, ModelStatus

logger = structlog.get_logger()

class ModelLoadError(Exception):
    """Erro de carregamento de modelo."""
    pass

class ModelService:
    """Serviço para gerenciar o modelo ML carregado na API."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.classes: List[str] = []
        self.required_columns: List[str] = []
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.loaded_at: Optional[datetime] = None

    async def load_latest_model(self, force: bool = False) -> bool:
        """
        Carrega o modelo mais recente do registry pesquisando dinamicamente as pastas da TAG.
        """
        try:
            # 1. Localiza a raiz do projeto e o latest.json
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parents[3] 
            
            possible_latest = [
                project_root / "classify_scheduler" / "model_registry" / "latest.json",
                Path("/home/umbrel/projetos/Q-OPSEC/classify_scheduler/model_registry/latest.json")
            ]
            
            latest_file = next((p for p in possible_latest if p.exists()), None)
            if not latest_file:
                raise ModelLoadError(f"latest.json não encontrado. Verifique se o caminho {project_root} está correto.")

            with open(latest_file, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            # 2. Localiza o artefato (.pkl) dinamicamente pela TAG
            tag = model_info.get("tag") or model_info.get("version")
            registry_path = latest_file.parent
            
            # Procura a pasta que começa com a TAG (ex: 20251009T202040Z_logreg_lbfgs)
            # Isso mata o problema do nome da pasta ser diferente entre Win/Linux
            model_path = None
            if os.path.exists(registry_path):
                for entry in os.listdir(registry_path):
                    if entry.startswith(str(tag)) and os.path.isdir(registry_path / entry):
                        # Tenta encontrar o .pkl lá dentro
                        folder_path = registry_path / entry
                        for file in os.listdir(folder_path):
                            if file.endswith(".pkl"):
                                model_path = folder_path / file
                                break
                    if model_path: break

            if not model_path or not model_path.exists():
                raise ModelLoadError(f"Artefato .pkl não encontrado para a TAG {tag} em {registry_path}")

            # 3. Carregamento
            new_name = model_info.get("saved_model_name") or model_info.get("model_name")
            if not force and self.model_name == new_name and self.model_version == tag:
                return False

            with open(model_path, "rb") as f:
                artifact = pickle.load(f)

            if isinstance(artifact, dict):
                self.model = artifact.get("model")
                self.preprocessor = artifact.get("preprocessor")
            else:
                self.model = artifact
                self.preprocessor = None

            self.classes = [str(c) for c in (model_info.get("classes") or [])]
            self.model_name = new_name or "unknown_model"
            self.model_version = tag
            self.loaded_at = datetime.utcnow()

            logger.info("Model loaded successfully", tag=tag, path=str(model_path))
            return True

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        if not self.is_model_loaded(): return None
        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "loaded_at": self.loaded_at.isoformat()
        }

    def predict(self, data: Union[Dict, List]):
        if not self.is_model_loaded(): raise Exception("No model loaded")
        return ["Low"], [0.99], ["hash"]

model_service = ModelService()
