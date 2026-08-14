# app/api/v1/endpoints_datasets.py
"""
Endpoints para gestão de datasets: upload, download, preview, schema, stats e validação.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import json
import os

from ...core.logging import get_logger
from ...core.security import get_current_user, require_auth
from ...services.model_service import model_service

logger = get_logger(__name__)
router = APIRouter()

# Configuração do diretório de datasets (usando raw string para Windows)
DATASETS_ROOT = Path(r"C:\Projetos\Q-OPSEC\classify_scheduler\datasets").resolve()
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Schemas
# ============================================================================

class DatasetInfo(BaseModel):
    """Informações de um dataset"""
    name: str = Field(..., description="Nome do dataset")
    path: str = Field(..., description="Path completo do dataset")
    total_size_bytes: int = Field(..., description="Tamanho total em bytes")
    files: List[str] = Field([], description="Lista de arquivos no dataset")
    modified_at: Optional[float] = Field(None, description="Timestamp da última modificação")


class CreateDatasetRequest(BaseModel):
    """Request para criar um novo dataset"""
    name: str = Field(..., min_length=1, description="Nome do dataset (será criado como diretório)")


# ============================================================================
# Dataset Management Endpoints
# ============================================================================

@router.get(
    "/datasets",
    response_model=List[DatasetInfo],
    tags=["Datasets"],
    summary="Lista todos os datasets",
    description="""
Lista todos os datasets disponíveis no diretório configurado.

Para cada dataset, retorna:
- Nome
- Path completo
- Tamanho total (soma de todos os arquivos)
- Lista de arquivos
- Timestamp da última modificação
""",
    responses={
        200: {
            "description": "Lista de datasets",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "fraud_detection_v1",
                            "path": "C:\\Projetos\\datasets\\fraud_detection_v1",
                            "total_size_bytes": 1048576,
                            "files": ["train.csv", "test.csv", "metadata.json"],
                            "modified_at": 1705324800.0
                        }
                    ]
                }
            }
        }
    }
)
async def list_datasets(user=Depends(get_current_user)):
    """
    Lista todos os datasets disponíveis.

    Varre o diretório DATASETS_ROOT e retorna informações de cada dataset.
    """
    datasets: List[DatasetInfo] = []
    for p in DATASETS_ROOT.iterdir():
        if p.is_dir():
            files = [f.name for f in p.iterdir() if f.is_file()]
            total = sum(f.stat().st_size for f in p.iterdir() if f.is_file())
            modified = max((f.stat().st_mtime for f in p.iterdir() if f.is_file()), default=p.stat().st_mtime)
            datasets.append(DatasetInfo(
                name=p.name,
                path=str(p),
                total_size_bytes=total,
                files=files,
                modified_at=modified
            ))
    return datasets


@router.post(
    "/datasets",
    response_model=DatasetInfo,
    tags=["Datasets"],
    summary="Cria um novo dataset",
    description="""
Cria um novo dataset (diretório vazio).

O nome do dataset será usado como nome do diretório.
Retorna erro 409 se o dataset já existir.

Requer autenticação.
""",
    responses={
        200: {"description": "Dataset criado com sucesso"},
        401: {"description": "Não autorizado"},
        409: {"description": "Dataset já existe"}
    }
)
async def create_dataset(payload: CreateDatasetRequest, user=Depends(require_auth)):
    """
    Cria um novo dataset (diretório).

    Args:
        payload: Nome do dataset
        user: Usuário autenticado

    Returns:
        Informações do dataset criado
    """
    target = DATASETS_ROOT / payload.name
    if target.exists():
        raise HTTPException(409, detail="Dataset already exists")
    target.mkdir(parents=True, exist_ok=False)
    logger.info("Dataset created", name=payload.name, path=str(target))
    return DatasetInfo(name=payload.name, path=str(target), total_size_bytes=0, files=[])


@router.get(
    "/datasets/{name}",
    response_model=DatasetInfo,
    tags=["Datasets"],
    summary="Informações de um dataset específico",
    description="Retorna informações detalhadas de um dataset pelo nome.",
    responses={
        200: {"description": "Informações do dataset"},
        404: {"description": "Dataset não encontrado"}
    }
)
async def get_dataset(name: str, user=Depends(get_current_user)):
    """
    Retorna informações de um dataset específico.

    Args:
        name: Nome do dataset
        user: Usuário autenticado

    Returns:
        Informações do dataset (arquivos, tamanho, etc.)
    """
    target = DATASETS_ROOT / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(404, detail="Dataset not found")
    files = [f.name for f in target.iterdir() if f.is_file()]
    total = sum(f.stat().st_size for f in target.iterdir() if f.is_file())
    modified = max((f.stat().st_mtime for f in target.iterdir() if f.is_file()), default=target.stat().st_mtime)
    return DatasetInfo(name=name, path=str(target), total_size_bytes=total, files=files, modified_at=modified)


@router.delete(
    "/datasets/{name}",
    status_code=204,
    tags=["Datasets"],
    summary="Deleta um dataset",
    description="""
Deleta um dataset e todos os seus arquivos.

**ATENÇÃO**: Esta operação é irreversível!

Requer autenticação.
""",
    responses={
        204: {"description": "Dataset deletado com sucesso"},
        400: {"description": "Path não é um diretório de dataset"},
        401: {"description": "Não autorizado"}
    }
)
async def delete_dataset(name: str, user=Depends(require_auth)):
    """
    Deleta um dataset e todos os seus arquivos.

    Args:
        name: Nome do dataset
        user: Usuário autenticado
    """
    target = DATASETS_ROOT / name
    if not target.exists():
        return JSONResponse(status_code=204, content=None)
    if not target.is_dir():
        raise HTTPException(400, detail="Not a dataset directory")

    # Delete recursivamente
    for p in target.rglob("*"):
        if p.is_file():
            p.unlink(missing_ok=True)
    for p in sorted(target.rglob("*"), reverse=True):
        if p.is_dir():
            p.rmdir()
    target.rmdir()

    logger.info("Dataset deleted", name=name)
    return JSONResponse(status_code=204, content=None)


# ============================================================================
# File Management Endpoints
# ============================================================================

@router.post(
    "/datasets/{name}/files",
    tags=["Datasets"],
    summary="Upload de arquivo para um dataset",
    description="""
Faz upload de um arquivo para um dataset.

- Se o arquivo já existir, use `override=true` para sobrescrever
- O arquivo é salvo com o nome original (`file.filename`)

Requer autenticação.
""",
    responses={
        200: {
            "description": "Arquivo enviado com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "filename": "train.csv",
                        "size_bytes": 1048576
                    }
                }
            }
        },
        404: {"description": "Dataset não encontrado"},
        409: {"description": "Arquivo já existe (use override=true)"},
        401: {"description": "Não autorizado"}
    }
)
async def upload_file(
        name: str,
        file: UploadFile = File(...),
        override: bool = Query(False, description="Sobrescrever se o arquivo já existir"),
        user=Depends(require_auth),
):
    """
    Faz upload de um arquivo para um dataset.

    Args:
        name: Nome do dataset
        file: Arquivo a ser enviado
        override: Se deve sobrescrever arquivo existente
        user: Usuário autenticado

    Returns:
        Status e informações do arquivo enviado
    """
    target_dir = DATASETS_ROOT / name
    if not target_dir.exists():
        raise HTTPException(404, detail="Dataset not found")

    dest = target_dir / file.filename
    if dest.exists() and not override:
        raise HTTPException(409, detail="File already exists. Use override=true")

    # Upload com arquivo temporário
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    tmp.replace(dest)

    logger.info("Dataset file uploaded", dataset=name, filename=file.filename, size_bytes=dest.stat().st_size)
    return {"status": "ok", "filename": file.filename, "size_bytes": dest.stat().st_size}


@router.get(
    "/datasets/{name}/files",
    tags=["Datasets"],
    summary="Lista arquivos de um dataset",
    description="Retorna lista de todos os arquivos em um dataset.",
    responses={
        200: {
            "description": "Lista de arquivos",
            "content": {
                "application/json": {
                    "example": {
                        "files": ["train.csv", "test.csv", "metadata.json"]
                    }
                }
            }
        },
        404: {"description": "Dataset não encontrado"}
    }
)
async def list_files(name: str, user=Depends(get_current_user)):
    """
    Lista todos os arquivos de um dataset.

    Args:
        name: Nome do dataset
        user: Usuário autenticado

    Returns:
        Lista de nomes de arquivos
    """
    target_dir = DATASETS_ROOT / name
    if not target_dir.exists():
        raise HTTPException(404, detail="Dataset not found")
    return {"files": [f.name for f in target_dir.iterdir() if f.is_file()]}


@router.get(
    "/datasets/{name}/files/{filename}",
    response_class=FileResponse,
    tags=["Datasets"],
    summary="Download de arquivo",
    description="Faz download de um arquivo específico de um dataset.",
    responses={
        200: {"description": "Arquivo para download"},
        404: {"description": "Arquivo não encontrado"}
    }
)
async def download_file(name: str, filename: str, user=Depends(get_current_user)):
    """
    Faz download de um arquivo de um dataset.

    Args:
        name: Nome do dataset
        filename: Nome do arquivo
        user: Usuário autenticado

    Returns:
        Arquivo para download
    """
    path = (DATASETS_ROOT / name / filename).resolve()
    if not path.exists() or not path.is_file() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    return FileResponse(path, filename=filename)


@router.delete(
    "/datasets/{name}/files/{filename}",
    status_code=204,
    tags=["Datasets"],
    summary="Deleta um arquivo",
    description="Deleta um arquivo específico de um dataset. Requer autenticação.",
    responses={
        204: {"description": "Arquivo deletado com sucesso"},
        401: {"description": "Não autorizado"}
    }
)
async def delete_file(name: str, filename: str, user=Depends(require_auth)):
    """
    Deleta um arquivo de um dataset.

    Args:
        name: Nome do dataset
        filename: Nome do arquivo
        user: Usuário autenticado
    """
    path = (DATASETS_ROOT / name / filename).resolve()
    if not path.exists() or not path.is_file() or DATASETS_ROOT not in path.parents:
        return JSONResponse(status_code=204, content=None)
    path.unlink(missing_ok=True)
    logger.info("Dataset file deleted", dataset=name, filename=filename)
    return JSONResponse(status_code=204, content=None)


# ============================================================================
# Metadata Endpoints
# ============================================================================

@router.post(
    "/datasets/{name}/metadata",
    tags=["Datasets"],
    summary="Define metadados do dataset",
    description="""
Define ou atualiza os metadados de um dataset.

Os metadados são salvos em `metadata.json` no diretório do dataset.

Requer autenticação.
""",
    responses={
        200: {"description": "Metadados salvos com sucesso"},
        404: {"description": "Dataset não encontrado"},
        401: {"description": "Não autorizado"}
    }
)
async def set_metadata(name: str, metadata: Dict[str, Any], user=Depends(require_auth)):
    """
    Define metadados de um dataset.

    Args:
        name: Nome do dataset
        metadata: Objeto JSON com metadados
        user: Usuário autenticado

    Returns:
        Status da operação
    """
    target = DATASETS_ROOT / name
    if not target.exists():
        raise HTTPException(404, detail="Dataset not found")
    with (target / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Dataset metadata updated", dataset=name)
    return {"status": "ok"}


@router.get(
    "/datasets/{name}/metadata",
    tags=["Datasets"],
    summary="Obtém metadados do dataset",
    description="Retorna os metadados de um dataset (arquivo metadata.json).",
    responses={
        200: {
            "description": "Metadados do dataset",
            "content": {
                "application/json": {
                    "example": {
                        "metadata": {
                            "description": "Dataset de detecção de fraude",
                            "version": "1.0",
                            "created_at": "2024-01-15"
                        }
                    }
                }
            }
        },
        404: {"description": "Dataset não encontrado"}
    }
)
async def get_metadata(name: str, user=Depends(get_current_user)):
    """
    Retorna metadados de um dataset.

    Args:
        name: Nome do dataset
        user: Usuário autenticado

    Returns:
        Metadados do dataset (ou objeto vazio se não existir)
    """
    target = DATASETS_ROOT / name
    if not target.exists():
        raise HTTPException(404, detail="Dataset not found")
    meta_path = target / "metadata.json"
    if not meta_path.exists():
        return {"metadata": {}}
    with meta_path.open("r", encoding="utf-8") as f:
        return {"metadata": json.load(f)}


# ============================================================================
# Data Analysis Endpoints (Preview, Schema, Stats)
# ============================================================================

def _read_table(path: Path, nrows: Optional[int] = None):
    """
    Lê um arquivo tabular (CSV, Parquet, JSONL) usando pandas.

    Args:
        path: Path do arquivo
        nrows: Número máximo de linhas a ler (None = todas)

    Returns:
        DataFrame do pandas
    """
    import pandas as pd
    if path.suffix.lower() in [".csv", ".tsv"]:
        return pd.read_csv(path, nrows=nrows)
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path) if nrows is None else pd.read_parquet(path).head(nrows)
    if path.suffix.lower() in [".jsonl", ".ndjson"]:
        return pd.read_json(path, lines=True, nrows=nrows)
    raise ValueError(f"Unsupported file type: {path.suffix}")


@router.get(
    "/datasets/{name}/preview",
    tags=["Datasets"],
    summary="Preview de dados",
    description="""
Retorna um preview (primeiras N linhas) de um arquivo tabular.

Suporta:
- CSV, TSV
- Parquet
- JSONL, NDJSON

Parâmetros:
- `file`: Nome do arquivo
- `n`: Número de linhas (padrão: 50, máx: 500)
""",
    responses={
        200: {
            "description": "Preview dos dados",
            "content": {
                "application/json": {
                    "example": {
                        "columns": ["id", "amount", "label"],
                        "rows": [
                            {"id": 1, "amount": 100.5, "label": "fraud"},
                            {"id": 2, "amount": 50.0, "label": "normal"}
                        ],
                        "count_returned": 2
                    }
                }
            }
        },
        400: {"description": "Erro ao ler arquivo (formato não suportado ou inválido)"},
        404: {"description": "Arquivo não encontrado"}
    }
)
async def preview(
        name: str,
        file: str = Query(..., description="Nome do arquivo"),
        n: int = Query(50, ge=1, le=500, description="Número de linhas"),
        user=Depends(get_current_user)
):
    """
    Retorna preview de um arquivo tabular.

    Args:
        name: Nome do dataset
        file: Nome do arquivo
        n: Número de linhas
        user: Usuário autenticado

    Returns:
        Colunas, linhas e contagem
    """
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        df = _read_table(path, nrows=n)
        return {
            "columns": list(df.columns),
            "rows": df.head(n).to_dict(orient="records"),
            "count_returned": min(n, len(df))
        }
    except Exception as e:
        logger.error("Preview error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Preview failed: {e}")


@router.get(
    "/datasets/{name}/schema",
    tags=["Datasets"],
    summary="Inferência de schema",
    description="""
Infere o schema de um arquivo tabular.

Retorna:
- Lista de colunas
- Tipos de dados (dtypes)
- Shape (linhas, colunas)
- Contagem de valores nulos por coluna
""",
    responses={
        200: {
            "description": "Schema inferido",
            "content": {
                "application/json": {
                    "example": {
                        "columns": ["id", "amount", "label"],
                        "dtypes": {"id": "int64", "amount": "float64", "label": "object"},
                        "shape": [1000, 3],
                        "null_counts": {"id": 0, "amount": 5, "label": 0}
                    }
                }
            }
        },
        400: {"description": "Erro ao inferir schema"},
        404: {"description": "Arquivo não encontrado"}
    }
)
async def schema(
        name: str,
        file: str = Query(..., description="Nome do arquivo"),
        user=Depends(get_current_user)
):
    """
    Infere o schema de um arquivo tabular.

    Args:
        name: Nome do dataset
        file: Nome do arquivo
        user: Usuário autenticado

    Returns:
        Schema com colunas, tipos, shape e null counts
    """
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        df = _read_table(path)
        inferred = {col: str(df[col].dtype) for col in df.columns}
        return {
            "columns": list(df.columns),
            "dtypes": inferred,
            "shape": list(df.shape),
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
        }
    except Exception as e:
        logger.error("Schema error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Schema inference failed: {e}")


@router.get(
    "/datasets/{name}/stats",
    tags=["Datasets"],
    summary="Estatísticas descritivas",
    description="""
Retorna estatísticas descritivas de um arquivo tabular.

Para colunas numéricas:
- Estatísticas descritivas (mean, std, min, max, quartis)

Para colunas categóricas:
- Top 10 valores mais frequentes
""",
    responses={
        200: {
            "description": "Estatísticas descritivas",
            "content": {
                "application/json": {
                    "example": {
                        "describe_numeric": {
                            "amount": {
                                "mean": 150.5,
                                "std": 50.2,
                                "min": 10.0,
                                "max": 500.0
                            }
                        },
                        "categorical_freqs": {
                            "label": {
                                "normal": 800,
                                "fraud": 200
                            }
                        }
                    }
                }
            }
        },
        400: {"description": "Erro ao calcular estatísticas"},
        404: {"description": "Arquivo não encontrado"}
    }
)
async def stats(
        name: str,
        file: str = Query(..., description="Nome do arquivo"),
        user=Depends(get_current_user)
):
    """
    Retorna estatísticas descritivas de um arquivo.

    Args:
        name: Nome do dataset
        file: Nome do arquivo
        user: Usuário autenticado

    Returns:
        Estatísticas numéricas e frequências categóricas
    """
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        import pandas as pd
        df = _read_table(path)
        desc_num = df.describe(include=["number"]).to_dict()
        # Frequências para categóricas (top 10)
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        cat_freqs = {c: df[c].value_counts().head(10).to_dict() for c in cat_cols}
        return {"describe_numeric": desc_num, "categorical_freqs": cat_freqs}
    except Exception as e:
        logger.error("Stats error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Stats failed: {e}")


# ============================================================================
# Validation Endpoint
# ============================================================================

@router.post(
    "/datasets/{name}/validate",
    tags=["Datasets"],
    summary="Valida dataset contra modelo",
    description="""
Valida se um arquivo de dataset é compatível com o modelo carregado.

Verifica:
- Se todas as colunas requeridas pelo modelo estão presentes
- Quais colunas estão faltando
- Quais colunas extras existem (não requeridas)

Usa uma amostra de 200 linhas para validação.
""",
    responses={
        200: {
            "description": "Resultado da validação",
            "content": {
                "application/json": {
                    "example": {
                        "valid": True,
                        "missing_columns": [],
                        "extra_columns": ["id", "timestamp"],
                        "required": ["feature1", "feature2", "feature3"]
                    }
                }
            }
        },
        400: {"description": "Erro ao validar"},
        404: {"description": "Arquivo não encontrado"},
        503: {"description": "Modelo não carregado"}
    }
)
async def validate_dataset(
        name: str,
        file: str = Query(..., description="Nome do arquivo"),
        user=Depends(get_current_user)
):
    """
    Valida dataset contra o modelo carregado.

    Args:
        name: Nome do dataset
        file: Nome do arquivo
        user: Usuário autenticado

    Returns:
        Resultado da validação (valid, missing_columns, extra_columns, required)
    """
    if not model_service.is_model_loaded():
        raise HTTPException(503, detail="Model not loaded")
    path = (DATASETS_ROOT / name / file).resolve()
    if not path.exists() or DATASETS_ROOT not in path.parents:
        raise HTTPException(404, detail="File not found")
    try:
        df = _read_table(path, nrows=200)  # sample
        required = set(model_service.get_required_columns() or [])
        missing = [c for c in required if c not in df.columns]
        extra = [c for c in df.columns if c not in required] if required else []
        valid = len(missing) == 0
        return {
            "valid": valid,
            "missing_columns": missing,
            "extra_columns": extra,
            "required": list(required)
        }
    except Exception as e:
        logger.error("Validation error", dataset=name, file=file, error=str(e))
        raise HTTPException(400, detail=f"Validation failed: {e}")