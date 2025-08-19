# services/risk_model_service.py
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
import os
import time
import joblib
import numpy as np
import pandas as pd

from models.schemas import AssessRequest, TrainRequest, RiskContext, TrainResponse
from repositories.config_repo import DATA_DIR, MODELS_DIR, set_best_model, get_best_model_info, read_registry, \
    write_registry, ConfigRepository

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

FEATURE_COLUMNS = [
    "global_alert_level_idx",
    "anomaly_index_global",
    "incident_rate_7d",
    "patch_delay_days_p50",
    "exposure_level_idx",
    "maintenance_window",
    "compliance_debt_score",
    "business_critical_period"
]

CATEGORICAL_MAPS = {
    "global_alert_level": {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4},
    "exposure_level": {"low": 0, "medium": 1, "high": 2},
}


def encode_row(signals: Dict[str, Any]) -> Dict[str, Any]:
    gal = signals.get("global_alert_level", "low")
    exp = signals.get("exposure_level", "medium")
    return {
        "global_alert_level_idx": CATEGORICAL_MAPS["global_alert_level"].get(str(gal).lower(), 1),
        "anomaly_index_global": float(signals.get("anomaly_index_global", 0.0)),
        "incident_rate_7d": int(signals.get("incident_rate_7d", 0)),
        "patch_delay_days_p50": int(signals.get("patch_delay_days_p50", 0)),
        "exposure_level_idx": CATEGORICAL_MAPS["exposure_level"].get(str(exp).lower(), 1),
        "maintenance_window": 1 if signals.get("maintenance_window", False) else 0,
        "compliance_debt_score": float(signals.get("compliance_debt_score", 0.0)),
        "business_critical_period": 1 if signals.get("business_critical_period", False) else 0,
    }


def synth_label(row: Dict[str, Any]) -> int:
    # Heurística simples para gerar um label consistente com risco
    score = (
            0.2 * row["global_alert_level_idx"]
            + 0.3 * row["anomaly_index_global"]
            + 0.2 * (row["incident_rate_7d"] / 100.0)
            + 0.1 * (row["patch_delay_days_p50"] / 30.0)
            + 0.15 * row["exposure_level_idx"]
            + 0.05 * row["compliance_debt_score"]
            + (0.1 if row["business_critical_period"] else 0.0)
            - (0.05 if row["maintenance_window"] else 0.0)
    )
    # 0/1/2 -> very_low/low/medium/high/critical mapearemos depois
    if score < 0.4:
        return 0
    elif score < 0.8:
        return 1
    else:
        return 2


class DatasetManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.train_csv = os.path.join(DATA_DIR, "signals_train.csv")
        self.valid_csv = os.path.join(DATA_DIR, "signals_valid.csv")

    def ensure_or_generate(self, n_train: int = 800, n_valid: int = 200, seed: int = 42):
        if os.path.exists(self.train_csv) and os.path.exists(self.valid_csv):
            return
        rng = np.random.default_rng(seed)

        def gen(n):
            rows = []
            for _ in range(n):
                s = {
                    "global_alert_level": np.random.choice(["none", "low", "medium", "high", "critical"],
                                                           p=[0.05, 0.35, 0.35, 0.2, 0.05]),
                    "current_campaigns_count": int(rng.integers(0, 5)),  # não usado no modelo baseline
                    "anomaly_index_global": float(rng.random()),
                    "incident_rate_7d": int(rng.integers(0, 200)),
                    "patch_delay_days_p50": int(rng.integers(0, 60)),
                    "exposure_level": np.random.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2]),
                    "maintenance_window": bool(rng.integers(0, 2)),
                    "compliance_debt_score": float(rng.random()),
                    "business_critical_period": bool(rng.integers(0, 2)),
                }
                enc = encode_row(s)
                label = synth_label(enc)
                rows.append({**s, **enc, "label": label})
            return pd.DataFrame(rows)

        train_df = gen(n_train)
        valid_df = gen(n_valid)
        train_df.to_csv(self.train_csv, index=False)
        valid_df.to_csv(self.valid_csv, index=False)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(self.train_csv)
        valid_df = pd.read_csv(self.valid_csv)
        return train_df, valid_df


class ModelCandidateFactory:
    @staticmethod
    def candidates(random_state: int = 42) -> List[Tuple[str, Any]]:
        models = []
        # Modelos clássicos
        models.append(("logreg", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ])))
        models.append(("rf", RandomForestClassifier(n_estimators=300, random_state=random_state)))
        models.append(("gb", GradientBoostingClassifier(random_state=random_state)))
        models.append(("mlp", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=random_state))
        ])))
        models.append(("linsvc", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(random_state=random_state))
        ])))

        if HAS_XGB:
            models.append(("xgb", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, n_jobs=2, eval_metric="mlogloss"
            )))
        if HAS_LGBM:
            models.append(("lgbm", LGBMClassifier(
                n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state
            )))
        return models


class ModelTrainer:
    def __init__(self, criterion: str = "accuracy"):
        assert criterion in ("accuracy", "f1"), "criterion must be accuracy or f1"
        self.criterion = criterion

    def train_and_select(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        results = []
        for name, model in ModelCandidateFactory.candidates():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average="macro")
                model_path = os.path.join(MODELS_DIR, f"model_{name}_{int(time.time())}.joblib")
                joblib.dump(model, model_path)
                results.append({"name": name, "path": model_path, "metrics": {"accuracy": acc, "f1": f1}})
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                continue

        if not results:
            raise Exception("No models trained successfully")

        # escolhe melhor
        key = (lambda r: r["metrics"]["accuracy"]) if self.criterion == "accuracy" else (lambda r: r["metrics"]["f1"])
        best = max(results, key=key)

        # atualiza registry
        registry = read_registry()
        all_models = registry.get("models", [])
        all_models.extend(results)
        registry["models"] = sorted(all_models, key=lambda r: r["metrics"]["accuracy"], reverse=True)
        write_registry(registry)
        set_best_model(best)
        return {"best": best, "all": results}


class RiskModelService:
    def __init__(self):
        self.dataset = DatasetManager()
        self.trainer = ModelTrainer(criterion="accuracy")
        self.cfg = ConfigRepository()
        self._best_model = None
        self._best_info = None
        self._retrain_lock = False
        self._load_best_from_disk()

    def _load_best_from_disk(self):
        info = get_best_model_info()
        if info and os.path.exists(info["path"]):
            try:
                self._best_model = joblib.load(info["path"])
                self._best_info = info
            except Exception:
                self._best_model = None
                self._best_info = None

    def _to_response(self, score: float, level: str, anomaly_score: float) -> RiskContext:
        now = datetime.utcnow()
        policies = self.cfg.get_policy_overrides(level)
        return RiskContext(
            score=score,
            level=level,
            anomaly_score=anomaly_score,
            threat_intel={"source": "ml"},
            recent_incidents=0,
            policy_overrides=policies,
            timestamp=now,
            model_version=self._best_info["name"] if self._best_info else "unknown"
        )

    def train(self, req: TrainRequest) -> TrainResponse:
        # Gera dataset se necessário
        n = max(50, min(5000, req.n))
        seed = req.seed if req.seed is not None else 42
        self.dataset.ensure_or_generate(n_train=int(0.8 * n), n_valid=int(0.2 * n), seed=seed)

        train_df, valid_df = self.dataset.load()
        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df["label"].values
        X_val = valid_df[FEATURE_COLUMNS].values
        y_val = valid_df["label"].values

        result = self.trainer.train_and_select(X_train, y_train, X_val, y_val)
        self._best_info = result["best"]
        self._best_model = joblib.load(self._best_info["path"])

        metrics = self._best_info["metrics"]
        return TrainResponse(
            model_version=f'{self._best_info["name"]}',
            metrics={"accuracy": float(metrics["accuracy"]), "f1": float(metrics["f1"])},
            samples=int(len(train_df) + len(valid_df))
        )

    def scheduled_retrain(self):
        if self._retrain_lock:
            return
        self._retrain_lock = True
        try:
            # Garante dataset; não altera seed aqui
            self.dataset.ensure_or_generate()
            train_df, valid_df = self.dataset.load()
            X_train = train_df[FEATURE_COLUMNS].values
            y_train = train_df["label"].values
            X_val = valid_df[FEATURE_COLUMNS].values
            y_val = valid_df["label"].values

            result = self.trainer.train_and_select(X_train, y_train, X_val, y_val)
            self._best_info = result["best"]
            self._best_model = joblib.load(self._best_info["path"])
        except Exception as e:
            print(f"Scheduled retrain failed: {e}")
        finally:
            self._retrain_lock = False

    def assess(self, req: AssessRequest) -> Optional[RiskContext]:
        if self._best_model is None:
            self._load_best_from_disk()
        if self._best_model is None:
            return None

        # Extrai features conforme GeneralSignals
        sig = req.signals.model_dump()
        enc = encode_row(sig)
        x = np.array([[enc[c] for c in FEATURE_COLUMNS]])
        pred = self._best_model.predict(x)[0]

        # map label -> score/level/anomaly
        label_to_level = {0: "low", 1: "medium", 2: "high"}
        level = label_to_level.get(int(pred), "low")
        score_map = {"low": 0.25, "medium": 0.55, "high": 0.8}
        anomaly = float(enc["anomaly_index_global"])

        return self._to_response(score=score_map[level], level=level, anomaly_score=anomaly)