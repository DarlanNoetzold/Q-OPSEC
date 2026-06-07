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
        Suporta ambientes mistos (Windows/Linux/WSL) e caminhos Linux fixos (Umbrel).
        """
        try:
            # 1. Tenta descobrir a raiz do projeto de forma dinâmica
            # Assume que estamos em classification_agent/app/services/model_service.py
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parents[3] # Sobe ate Q-OPSEC
            
            # Lista de caminhos possíveis para o latest.json baseados no ambiente do usuário
            possible_latest_paths = [
                project_root / "classify_scheduler" / "model_registry" / "latest.json",
                Path("/home/umbrel/projetos/Q-OPSEC/classify_scheduler/model_registry/latest.json"),
                Path("/mnt/c/Projetos/Q-OPSEC/classify_scheduler/model_registry/latest.json"),
                Path(settings.ml_registry_dir) / settings.ml_registry_latest_file
            ]
            
            latest_file = None
            for p in possible_latest_paths:
                if p.exists():
                    latest_file = p
                    break
            
            if not latest_file:
                raise ModelLoadError(f"Arquivo latest.json não encontrado. Tentativas: {[str(p) for p in possible_latest_paths]}")
            
            with open(latest_file, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            # 2. Resolver o path do artefato (.pkl)
            # O JSON costuma vir com path de Windows (C:\Projetos\...)
            raw_artifact_path = model_info.get("artifact_path") or model_info.get("file_path")
            if not raw_artifact_path:
                raise ModelLoadError("O arquivo latest.json não contém caminhos para o artefato do modelo.")

            # Normaliza barras (Windows \ -> Linux /)
            normalized_artifact_name = Path(raw_artifact_path.replace("\\", "/")).name
            artifact_tag = model_info.get("tag") or model_info.get("version")
            
            # Tenta reconstruir o path no Linux de forma resiliente
            # Padrão: /home/umbrel/projetos/Q-OPSEC/classify_scheduler/model_registry/{tag}/model.pkl
            candidate_artifact_paths = [
                # Tentativa 1: Pasta da TAG + nome do arquivo original (mais comum)
                project_root / "classify_scheduler" / "model_registry" / artifact_tag / normalized_artifact_name,
                # Tentativa 2: Pasta da TAG + model.pkl (padrao fixo)
                project_root / "classify_scheduler" / "model_registry" / artifact_tag / "model.pkl",
                # Tentativa 3: Path absoluto forçado para Umbrel
                Path(f"/home/umbrel/projetos/Q-OPSEC/classify_scheduler/model_registry/{artifact_tag}/model.pkl"),
                # Tentativa 4: Path relativo direto
                project_root.parent / "classify_scheduler" / "model_registry" / artifact_tag / "model.pkl"
            ]

            model_path = None
            for p in candidate_artifact_paths:
                if p.exists():
                    model_path = p
                    break

            if not model_path:
                raise ModelLoadError(f"Artefato (.pkl) não encontrado. Tentativas: {[str(p) for p in candidate_artifact_paths]}")

            # 3. Carregamento do Pickle
            new_name = model_info.get("saved_model_name") or model_info.get("model_name")
            new_version = artifact_tag

            if not force and self.model_name == new_name and self.model_version == new_version:
                return False

            with open(model_path, "rb") as f:
                artifact = pickle.load(f)

            if isinstance(artifact, dict):
                self.model = artifact.get("model")
                self.preprocessor = artifact.get("preprocessor")
            else:
                self.model = artifact
                self.preprocessor = None

            # 4. Metadados
            self.classes = [str(c) for c in (model_info.get("classes") or [])]
            if not self.classes and hasattr(self.model, "classes_"):
                self.classes = [str(c) for c in self.model.classes_]

            self.required_columns = list(model_info.get("required_columns") or [])
            self.model_name = new_name or "unknown_model"
            self.model_version = new_version or "unknown_version"
            self.loaded_at = datetime.utcnow()

            logger.info("Model loaded successfully", 
                        environment="UMBREL_COMPATIBLE", 
                        path=str(model_path))
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
            "model_name": self.model_name,
            "version": self.model_version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "hash": hashlib.md5(str(self.model_version).encode()).hexdigest()
        }

    def predict(self, data: Union[Dict, List]):
        if not self.is_model_loaded():
            raise PredictionError("No model loaded")
        # Logica de predicao sera executada aqui
        return ["Low"], [0.99], ["hash"]

model_service = ModelService()
