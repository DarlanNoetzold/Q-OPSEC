from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, f1_score

from preprocessing import build_preprocessor
from models_zoo import make_models

@dataclass
class TrainResult:
    best_name: str
    best_cv_metrics: Dict[str, float]
    pipeline: Any
    label_encoder: LabelEncoder
    X_test: pd.DataFrame
    y_test: np.ndarray

def _scorers():
    return {
        "accuracy": make_scorer(accuracy_score),
        "f1_macro": make_scorer(f1_score, average="macro"),
    }

def train_and_select_best(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    seed: int,
    n_splits: int,
    scoring_primary: str,
    scoring_secondary: str,
    max_models: int = 20
) -> TrainResult:
    df = df.copy()
    y_raw = df[target_col].astype(str).values
    X = df.drop(columns=[target_col])

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pre, feat_cols = build_preprocessor(X_train, target_col=None)
    models = make_models(pre)
    names = list(models.keys())[:max_models]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorers = _scorers()

    best_name = None
    best_scores = None
    best_estimator = None

    for name in names:
        model = models[name]
        cv = cross_validate(model, X_train, y_train, cv=kf, scoring=scorers, n_jobs=-1, return_train_score=False)
        acc = float(np.mean(cv["test_accuracy"]))
        f1m = float(np.mean(cv["test_f1_macro"]))
        print(f"[CV] {name}: accuracy={acc:.4f} f1_macro={f1m:.4f}")

        if best_scores is None:
            best_name, best_scores, best_estimator = name, {"accuracy": acc, "f1_macro": f1m}, model
        else:
            ba, bf = best_scores["accuracy"], best_scores["f1_macro"]
            if (acc > ba) or (acc == ba and f1m > bf):
                best_name, best_scores, best_estimator = name, {"accuracy": acc, "f1_macro": f1m}, model

    best_estimator.fit(X_train, y_train)

    return TrainResult(
        best_name=best_name,
        best_cv_metrics=best_scores,
        pipeline=best_estimator,
        label_encoder=le,
        X_test=X_test,
        y_test=y_test
    )