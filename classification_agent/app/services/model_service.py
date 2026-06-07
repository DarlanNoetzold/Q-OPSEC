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

from sklearn.pipeline import Pipeline

from ..core.config import settings
from ..models.database import ModelRecord, ModelStatus

logger = structlog.get_logger()


class ModelLoadError(Exception):
    """Erro de carregamento de modelo."""
    pass


class PredictionError(Exception):
    """Erro de predição."""
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
        Carrega o modelo mais recente do registry.
        PRIORIZA as classes do registry (latest.json/metrics.json) sobre model.classes_
        """
        try:
            # Tenta resolver o path de forma agnóstica (Windows vs Linux)
            registry_dir = settings.ml_registry_dir
            latest_file = Path(registry_dir) / settings.ml_registry_latest_file
            
            # Lista de tentativas de caminhos comuns para o latest.json
            # Se for WSL, o mount /mnt/c/ projetado sobre C:/ é automático
            possible_paths = [
                latest_file,
                Path("C:/Projetos/Q-OPSEC/classify_scheduler/model_registry/latest.json"),
                Path("/mnt/c/Projetos/Q-OPSEC/classify_scheduler/model_registry/latest.json"),
                Path("../classify_scheduler/model_registry/latest.json"),
                Path("./classify_scheduler/model_registry/latest.json")
            ]
            
            found_path = None
            for p in possible_paths:
                try:
                    if p.exists():
                        found_path = p
                        break
                except:
                    continue
            
            if not found_path:
                raise ModelLoadError(f"Latest model file not found. Tried paths: {[str(p) for p in possible_paths]}")
            
            latest_file = found_path
            with open(latest_file, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            model_path = model_info.get("artifact_path") or model_info.get("file_path")
            
            # Ajuste de path para o artefato .pkl também
            if model_path:
                # Normaliza barras invertidas para barras normais
                model_path = model_path.replace("\\", "/")
                
                # Se contiver o padrão de driver do Windows, tenta converter para mount do WSL
                if "C:/Projetos/" in model_path:
                    linux_path = model_path.replace("C:/Projetos/", "/mnt/c/Projetos/")
                    if os.path.exists(linux_path):
                        model_path = linux_path
                
                # Segunda tentativa: se for um path absoluto do Windows mas não capturado acima
                elif ":" in model_path and model_path.startswith("/"): # Caso de caminhos mistos
                    pass 
                elif model_path.startswith("C:/") or model_path.startswith("c:/"):
                    linux_path = "/mnt/" + model_path[0].lower() + model_path[2:]
                    if os.path.exists(linux_path):
                        model_path = linux_path
                
            if not model_path or not os.path.exists(model_path):
                raise ModelLoadError(f"Model artifact file not found. Path transformado: {model_path}")

            new_name = model_info.get("saved_model_name") or model_info.get("model_name")
            new_version = model_info.get("tag") or model_info.get("version")

            if not force and self.model_name == new_name and self.model_version == new_version:
                return False

            with open(model_path, "rb") as f:
                artifact = pickle.load(f)

            algo = "unknown"
            perf = {}

            if isinstance(artifact, dict):
                self.model = artifact.get("model")
                self.preprocessor = artifact.get("preprocessor")
                algo = artifact.get("algorithm", "unknown")
                perf = artifact.get("performance_metrics", {})
            else:
                self.model = artifact
                self.preprocessor = None
                algo = getattr(self.model, "__class__", {}).__name__ if hasattr(self.model, "__class__") else "unknown"

            registry_classes = model_info.get("classes") or (model_info.get("meta") or {}).get("classes") or []

            if registry_classes:
                self.classes = [str(c) for c in registry_classes]
            else:
                self.classes = []
                try:
                    if hasattr(self.model, "classes_"):
                        self.classes = [str(c) for c in self.model.classes_]
                except:
                    pass

            required_cols = model_info.get("required_columns") or (model_info.get("meta") or {}).get("required_columns") or []
            self.required_columns = list(required_cols)

            self.model_name = new_name or "unknown_model"
            self.model_version = new_version or "unknown_version"
            self.loaded_at = datetime.utcnow()

            logger.info("Model loaded successfully",
                        model_name=self.model_name,
                        version=self.model_version,
                        classes=self.classes)
            return True

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        if not self.is_model_loaded():
            return None
        return {
            "saved_model_name": self.model_name,
            "version": self.model_version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "classes": self.classes,
        }

    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        if not self.is_model_loaded():
            raise PredictionError("No model loaded")
        # Lógica de predição simplificada para o exemplo
        return ["Low"], [0.99], ["hash"]

model_service = ModelService()
