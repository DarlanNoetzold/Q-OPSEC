import re
import numpy as np
import random
import pandas as pd
from typing import List, Tuple, Dict
import torch

CANONICAL = {
    "very low": "Very Low",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "very high": "Very High",
    "critical": "Critical",
}

def to_canonical(label: str) -> str:
    if not isinstance(label, str):
        return ""
    key = label.strip().lower()
    return CANONICAL.get(key, "")

def normalize_labels(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    df[target_col] = df[target_col].apply(to_canonical)
    before = len(df)
    df = df[df[target_col] != ""]
    return df, before - len(df)

def ensure_classes(df: pd.DataFrame, target_col: str, allowed: List[str]):
    present = sorted(df[target_col].unique().tolist())
    missing = [c for c in allowed if c not in present]
    if missing:
        print(f"[WARN] Missing target classes in dataset: {missing}")

def print_summary_classes(df: pd.DataFrame, target_col: str):
    print("[DATA] Class distribution:")
    print(df[target_col].value_counts(normalize=True).round(3))

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)  # opcional, se torch estiver instalado
    except Exception:
        pass

def available(lib_name: str) -> bool:
    try:
        __import__(lib_name)
        return True
    except Exception:
        return False