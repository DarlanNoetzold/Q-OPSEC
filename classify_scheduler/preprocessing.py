import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

DROP_COLS = [
    "id", "request_id_resolved", "created_at",
    "encryption_script_label", "processing_priority_label"
]

TARGET_FALLBACK = "security_level_label"

def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, List[str]]:
    candidate_cols = [c for c in df.columns if c not in DROP_COLS + [target_col]]
    num_cols = []
    cat_cols = []
    for c in candidate_cols:
        if pd.api.types.is_bool_dtype(df[c]) or df[c].dropna().isin([0,1]).all():
            num_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False))
    ])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    feature_cols = num_cols + cat_cols
    return pre, feature_cols