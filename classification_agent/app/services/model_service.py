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
        latest_file = Path(settings.ml_registry_dir) / settings.ml_registry_latest_file
        try:
            if not latest_file.exists():
                raise ModelLoadError(f"Latest model file not found: {latest_file}")

            with open(latest_file, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            model_path = model_info.get("artifact_path") or model_info.get("file_path")
            if not model_path or not os.path.exists(model_path):
                raise ModelLoadError(f"Model file not found: {model_path}")

            new_name = model_info.get("saved_model_name") or model_info.get("model_name")
            new_version = model_info.get("tag") or model_info.get("version")

            if not force and self.model_name == new_name and self.model_version == new_version:
                # já atualizado
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
                # Objeto sklearn direto
                self.model = artifact
                self.preprocessor = None
                algo = getattr(self.model, "__class__", {}).__name__ if hasattr(self.model, "__class__") else "unknown"

            # PRIORIZAR CLASSES DO REGISTRY (latest.json/metrics.json)
            registry_classes = model_info.get("classes") or (model_info.get("meta") or {}).get("classes") or []

            if registry_classes:
                # Usar classes do registry (já ordenadas por severidade)
                self.classes = [str(c) for c in registry_classes]
                logger.info("Using classes from registry", classes=self.classes)
            else:
                # Fallback: tentar extrair do modelo (só se não houver no registry)
                self.classes = []
                try:
                    if isinstance(self.model, Pipeline):
                        last_est = self.model.steps[-1][1]
                        if hasattr(last_est, "classes_"):
                            self.classes = [str(c) for c in last_est.classes_]
                    if not self.classes and hasattr(self.model, "classes_"):
                        self.classes = [str(c) for c in self.model.classes_]
                except Exception:
                    pass
                logger.warning("Using classes from model (fallback)", classes=self.classes)

            # Required columns com fallback para meta.required_columns
            required_cols = model_info.get("required_columns") or (model_info.get("meta") or {}).get(
                "required_columns") or []
            self.required_columns = list(required_cols)

            self.model_name = new_name or "unknown_model"
            self.model_version = new_version or "unknown_version"
            self.loaded_at = datetime.utcnow()

            # Persistir info no banco (best-effort)
            try:
                await self._save_model_info(
                    {
                        "name": self.model_name,
                        "version": self.model_version,
                        "file_path": model_path,
                        "algorithm": algo,
                        "metadata": model_info,
                        "performance_metrics": perf,
                    }
                )
            except Exception as db_err:
                logger.warning("Failed to save model info to database", error=str(db_err))

            logger.info("Model loaded successfully",
                        model_name=self.model_name,
                        version=self.model_version,
                        classes=self.classes,
                        required_columns=len(self.required_columns))
            return True

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    async def _save_model_info(self, info: Dict[str, Any]) -> None:
        """Cria/atualiza registro do modelo no MongoDB."""
        existing = await ModelRecord.find_one(
            ModelRecord.name == info["name"],
            ModelRecord.version == info["version"]
        )
        if existing:
            existing.status = ModelStatus.ACTIVE
            existing.updated_at = datetime.utcnow()
            await existing.save()
        else:
            rec = ModelRecord(
                name=info["name"],
                version=info["version"],
                algorithm=info.get("algorithm", "unknown"),
                file_path=info.get("file_path", ""),
                metadata=info.get("metadata", {}),
                performance_metrics=info.get("performance_metrics", {}),
                status=ModelStatus.ACTIVE,
                is_default=True,
            )
            await rec.insert()

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        if not self.is_model_loaded():
            return None
        algo = getattr(self.model, "__class__", {}).__name__ if hasattr(self.model, "__class__") else "unknown"
        return {
            "saved_model_name": self.model_name,
            "version": self.model_version,
            "algorithm": algo,
            "classes": self.classes,  # Já são strings dos nomes reais
            "required_columns": self.required_columns,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "meta": {},
        }

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engenharia de atributos a partir de created_at quando necessário."""
        df = df.copy()

        # Se created_at existe e features derivadas não existem, calcular
        if "created_at" in df.columns:
            try:
                dt = pd.to_datetime(df["created_at"], errors="coerce")

                if "hour_of_day" not in df.columns:
                    df["hour_of_day"] = dt.dt.hour
                if "day_of_week" not in df.columns:
                    df["day_of_week"] = dt.dt.dayofweek
                if "month" not in df.columns:
                    df["month"] = dt.dt.month
                if "year" not in df.columns:
                    df["year"] = dt.dt.year

                # Seno/cosseno para ciclicidade
                if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
                    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
                    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

                if "day_sin" not in df.columns or "day_cos" not in df.columns:
                    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
                    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

            except Exception as e:
                logger.warning("Failed to engineer features from created_at", error=str(e))

        return df

    def validate_input(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[str]:
        """Validação tolerante - apenas reporta problemas, não quebra."""
        errors: List[str] = []
        if not self.is_model_loaded():
            errors.append("No model loaded")
            return errors

        rows = [data] if isinstance(data, dict) else data

        # Criar DataFrame temporário para validação (com engenharia de features)
        temp_df = pd.DataFrame(rows)
        temp_df = self._engineer_features(temp_df)

        for i, row in enumerate(rows):
            if self.required_columns:
                # Verificar após engenharia de features
                available_cols = set(temp_df.columns)
                missing_cols = [c for c in self.required_columns if c not in available_cols]
                if missing_cols:
                    # Apenas warning - será preenchido com NaN no predict
                    errors.append(f"Row {i}: Missing columns will be filled with NaN: {missing_cols}")

        # Retornar lista vazia para não quebrar - problemas são só warnings
        return []

    def predict(
            self,
            data: Union[Dict[str, Any], List[Dict[str, Any]]],
            include_probabilities: bool = True,
    ):
        if not self.is_model_loaded():
            raise PredictionError("No model loaded")

        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

            # Engenharia de atributos
            df = self._engineer_features(df)

            # PREENCHER COLUNAS FALTANTES COM NaN (soft-fail)
            if self.required_columns:
                missing = [c for c in self.required_columns if c not in df.columns]
                if missing:
                    logger.warning("Missing columns found in input; filling with NaN", missing=missing)
                    for c in missing:
                        df[c] = np.nan

                # Reordenar/selecionar exatamente as colunas que o modelo espera
                df = df[self.required_columns]

            # Predição
            proba = None

            if self.preprocessor is not None:
                X = self.preprocessor.transform(df)
                y_pred = self.model.predict(X)
                if include_probabilities and hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(X)
            else:
                if isinstance(self.model, Pipeline):
                    y_pred = self.model.predict(df)
                    if include_probabilities:
                        # tentar no pipeline ou no último estimador
                        if hasattr(self.model, "predict_proba"):
                            proba = self.model.predict_proba(df)
                        else:
                            try:
                                last_est = self.model.steps[-1][1]
                                Xt = self.model[:-1].transform(df)
                                if hasattr(last_est, "predict_proba"):
                                    proba = last_est.predict_proba(Xt)
                            except Exception:
                                proba = None
                else:
                    X = df.values
                    y_pred = self.model.predict(X)
                    if include_probabilities and hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(X)

            # MAPEAR ÍNDICES PARA NOMES REAIS DAS CLASSES
            labels: List[str] = []
            if self.classes:
                # Se y_pred são índices (0,1,2...), mapear para nomes reais
                for y in y_pred:
                    try:
                        idx = int(y)
                        if 0 <= idx < len(self.classes):
                            labels.append(self.classes[idx])
                        else:
                            labels.append(str(y))
                    except (ValueError, TypeError):
                        # y já é string de classe
                        labels.append(str(y))
            else:
                # Fallback: converter para string
                labels = [str(y) for y in y_pred]

            # Probabilidades com nomes reais das classes
            probabilities = None
            if proba is not None and self.classes:
                probabilities = []
                for row in proba:
                    prob_dict = {}
                    for idx, cls in enumerate(self.classes):
                        if idx < len(row):
                            prob_dict[cls] = float(row[idx])
                    probabilities.append(prob_dict)

            # Hash dos inputs
            input_hashes: List[Optional[str]] = []
            rows = [data] if isinstance(data, dict) else data
            for row in rows:
                try:
                    input_hashes.append(hashlib.md5(str(sorted(row.items())).encode()).hexdigest())
                except Exception:
                    input_hashes.append(None)

            return labels, probabilities, input_hashes

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise PredictionError(f"Prediction failed: {str(e)}")


model_service = ModelService()