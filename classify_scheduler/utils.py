import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

# Mapeamento canônico para normalização de labels
CANONICAL = {
    "very low": "Very Low",
    "verylow": "Very Low",
    "very_low": "Very Low",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "very high": "Very High",
    "veryhigh": "Very High",
    "very_high": "Very High",
    "critical": "Critical",
}


def available(lib_name: str) -> bool:
    """Verifica se uma biblioteca está disponível para importação."""
    try:
        __import__(lib_name)
        return True
    except Exception:
        return False


def to_canonical(label: str) -> str:
    """Converte um label para sua forma canônica."""
    if not isinstance(label, str):
        return ""
    key = label.strip().lower()
    return CANONICAL.get(key, "")


def seed_everything(seed: int):
    """Fixa seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def normalize_labels(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, int]:
    """
    Normaliza rótulos usando o mapeamento CANONICAL.
    Retorna (df_limpo, num_dropped_rows).
    """
    df = df.copy()
    df[target_col] = df[target_col].apply(to_canonical)
    before = len(df)
    df = df[df[target_col] != ""]
    return df, before - len(df)


def ensure_classes(df: pd.DataFrame, target_col: str, allowed: List[str]):
    """Garante que o dataset tem pelo menos 2 classes válidas."""
    present = sorted(df[target_col].unique().tolist())
    missing = [c for c in allowed if c not in present]
    if missing:
        print(f"[WARN] Missing target classes in dataset: {missing}")
    if len(present) < 2:
        raise ValueError(f"Dataset has only {len(present)} class(es): {present}. Need at least 2 classes.")


def print_summary_classes(df: pd.DataFrame, target_col: str):
    """Imprime resumo das classes no dataset."""
    print("[DATA] Class distribution:")
    vc = df[target_col].value_counts()
    pv = df[target_col].value_counts(normalize=True).round(3)
    for cls in vc.index:
        print(f"  {cls}: {vc[cls]} ({pv[cls]:.1%})")


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """Valida se o DataFrame tem todas as colunas obrigatórias."""
    missing = [col for col in required_cols if col not in df.columns]
    return missing


def create_feature_mapping() -> Dict[str, Any]:
    """Cria mapeamento de features para documentação."""
    return {
        "temporal_features": [
            "hour_of_day", "day_of_week", "month", "year",
            "hour_sin", "hour_cos", "day_sin", "day_cos"
        ],
        "risk_features": [
            "risk_score", "conf_score", "combined_score"
        ],
        "categorical_features": [
            "risk_level", "conf_classification", "src_geo",
            "src_device_type", "dst_service_type", "dst_security_policy",
            "src_mfa_status_norm"
        ],
        "identifier_features": [
            "id", "request_id_resolved"
        ]
    }


def get_feature_columns_from_df(df: pd.DataFrame, target_col: str, exclude_identifiers: bool = True) -> List[str]:
    """
    Extrai colunas de features do DataFrame, excluindo target e opcionalmente identificadores.
    """
    all_cols = list(df.columns)
    feature_cols = [col for col in all_cols if col != target_col]

    if exclude_identifiers:
        # Excluir colunas que parecem identificadores
        identifier_patterns = ["id", "request_id", "_id"]
        feature_cols = [
            col for col in feature_cols
            if not any(pattern in col.lower() for pattern in identifier_patterns)
        ]

    return feature_cols


def log_model_performance(metrics: Dict[str, float], model_name: str):
    """Log formatado das métricas do modelo."""
    print(f"[PERFORMANCE] {model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def format_class_mapping(classes: List[str]) -> Dict[int, str]:
    """Cria mapeamento índice -> nome da classe."""
    return {i: cls for i, cls in enumerate(classes)}


def validate_security_levels(levels: List[str]) -> bool:
    """Valida se os níveis de segurança estão na ordem correta."""
    from configs import ALLOWED_CLASSES
    expected_order = ALLOWED_CLASSES
    return levels == expected_order[:len(levels)]