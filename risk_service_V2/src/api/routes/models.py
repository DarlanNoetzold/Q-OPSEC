from typing import Optional, Any, Dict
from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path
import json
import traceback

from src.api.models.model_manager import ModelManager
from src.common.logger import logger

router = APIRouter()

_manager: Optional[ModelManager] = None

def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        logger.info("Initializing ModelManager instance in get_manager()")
        _manager = ModelManager()
        try:
            _manager.load()
            logger.info(f"ModelManager loaded version: {_manager.loaded_version}")
            logger.debug(f"Feature names loaded: {_manager.feature_names[:20] if isinstance(_manager.feature_names, list) else str(_manager.feature_names)[:200]}")
        except Exception as e:
            logger.exception("Failed to load models in get_manager()")
            raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")
    else:
        logger.debug("Using cached ModelManager instance in get_manager()")
    return _manager

@router.get("/feature_names")
async def get_feature_names(manager: ModelManager = Depends(get_manager)):
    """
    Rota principal - retorna resumo do atributo feature_names carregado.
    """
    logger.info("GET /models/feature_names called")
    fn = manager.feature_names
    try:
        sample = fn[:200] if isinstance(fn, (list, tuple)) else (list(fn.keys())[:200] if isinstance(fn, dict) else str(fn))
        logger.debug(f"feature_names type: {type(fn)}; sample: {sample}")
        response = {
            "version": manager.loaded_version,
            "feature_names_type": str(type(fn)),
            "n_feature_names": len(fn) if hasattr(fn, "__len__") else None,
            "feature_names_sample": sample
        }
        logger.info(f"Returning feature_names summary with {response['n_feature_names']} features")
        return response
    except Exception as e:
        logger.exception("Error while preparing feature_names response")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature_names/raw")
async def get_feature_names_raw():
    """
    DEBUG endpoint: retorna o conteúdo bruto do arquivo feature_names.json do último modelo.
    Útil para checar se o arquivo contém uma dict em vez de uma lista.
    """
    logger.info("GET /models/feature_names/raw called")
    try:
        mm = ModelManager()
        ver = None
        try:
            ver_dir = mm._get_latest_version_dir()
            ver = ver_dir.name if ver_dir else None
            logger.debug(f"Latest model version directory: {ver}")
        except Exception as e:
            logger.warning(f"Failed to get latest version dir: {e}")
            ver = None

        base = Path(mm.models_root)
        if ver:
            fpath = base / ver / "feature_names.json"
        else:
            latest = mm._get_latest_version_dir()
            if latest is None:
                logger.error("No model versions found on disk")
                return {"error": "No model versions found on disk", "models_root": str(base)}
            fpath = latest / "feature_names.json"

        logger.debug(f"Looking for feature_names.json at: {fpath}")

        if not fpath.exists():
            logger.error(f"feature_names.json not found at {fpath}")
            return {"error": "feature_names.json not found", "checked_path": str(fpath)}

        raw_text = fpath.read_text(encoding="utf-8")
        try:
            raw = json.loads(raw_text)
            logger.debug(f"feature_names.json loaded, type: {type(raw)}")
        except Exception as e:
            logger.warning(f"Failed to parse feature_names.json as JSON: {e}")
            raw = {"_raw_text": raw_text}

        return {"path": str(fpath), "version": ver, "raw_type": str(type(raw)), "raw_sample": (raw if isinstance(raw, (list, dict)) else str(raw))}
    except Exception as e:
        logger.exception("Error in feature_names/raw")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature_names/debug_columns")
async def get_feature_names_debug_columns():
    """
    DEBUG endpoint: carrega manager em memória, e retorna
    - tipo de feature_names em memória
    - primeiras 200 feature_names
    - se possível, um snapshot de colunas de um DataFrame vazio criado usando feature_names
    - e um exemplo de índice que causaria o erro "Passing a dict as an indexer"
    """
    logger.info("GET /models/feature_names/debug_columns called")
    try:
        manager = get_manager()
        fn = manager.feature_names
        info: Dict[str, Any] = {
            "version": manager.loaded_version,
            "feature_names_type": str(type(fn)),
            "n_feature_names": len(fn) if hasattr(fn, "__len__") else None,
            "feature_names_sample": fn[:200] if isinstance(fn, (list, tuple)) else (list(fn.keys())[:200] if isinstance(fn, dict) else str(fn))
        }
        logger.debug(f"feature_names info: {info}")

        # tenta criar DataFrame vazio e indexar para reproduzir erro
        try:
            import pandas as pd, numpy as np
            df = pd.DataFrame([{}])  # one-row empty
            logger.debug("Created empty DataFrame for testing")

            # ensure to coerce feature_names to list for the test
            if not isinstance(fn, list):
                try:
                    test_fn = fn.get("all_features") if isinstance(fn, dict) and "all_features" in fn else list(fn)
                    logger.debug("Coerced feature_names dict to list for test")
                except Exception as e:
                    logger.warning(f"Failed to coerce feature_names to list: {e}")
                    test_fn = list(fn or [])
            else:
                test_fn = fn

            # add missing columns
            for c in (test_fn or []):
                if c not in df.columns:
                    df[c] = np.nan
            logger.debug(f"Added missing columns to DataFrame: {len(test_fn)} columns")

            # attempt indexing (this is where dict-as-indexer would crash)
            try:
                df_indexed = df[test_fn]  # if test_fn is dict -> triggers the same error
                info["df_indexed_shape"] = df_indexed.shape
                logger.info(f"DataFrame indexed successfully with shape {df_indexed.shape}")
            except Exception as e:
                info["indexing_error"] = str(e)
                info["indexing_traceback"] = traceback.format_exc()
                info["df_columns_sample"] = df.columns.tolist()[:200]
                info["test_indexer_sample_type"] = str(type(test_fn))
                logger.error(f"Indexing error: {e}")
                logger.debug(traceback.format_exc())
        except Exception as e:
            info["df_creation_error"] = str(e)
            info["df_creation_traceback"] = traceback.format_exc()
            logger.error(f"DataFrame creation error: {e}")
            logger.debug(traceback.format_exc())

        return info
    except Exception as e:
        logger.exception("Error in feature_names/debug_columns")
        raise HTTPException(status_code=500, detail=str(e))