import threading
import random
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from models.schemas import GeneralSignals, AssessRequest, RiskContext, TrainRequest, TrainResponse
from repositories.config_repo import ConfigRepository

SEVERITY_WEIGHTS = {"none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.75, "critical": 1.0}
EXPOSURE_WEIGHTS = {"low": 0.2, "medium": 0.5, "high": 0.8}

def campaigns_severity_max(campaigns: List[Dict[str, str]]) -> float:
    sev = 0.0
    for c in campaigns or []:
        sev = max(sev, SEVERITY_WEIGHTS.get(c.get("severity", "medium"), 0.5))
    return sev

def to_features(s: GeneralSignals):
    base_alert = SEVERITY_WEIGHTS.get(s.global_alert_level, 0.5)
    campaign_sev = campaigns_severity_max(s.current_campaigns)
    anomaly = np.clip(s.anomaly_index_global, 0.0, 1.0)
    incidents = s.incident_rate_7d
    patch = min(1.0, s.patch_delay_days_p50 / 30.0)
    exposure = EXPOSURE_WEIGHTS.get(s.exposure_level, 0.5)
    compliance = s.compliance_debt_score
    maint = 1.0 if s.maintenance_window else 0.0
    critical_period = 1.0 if s.business_critical_period else 0.0

    x = np.array([
        base_alert, campaign_sev, anomaly, incidents, patch,
        exposure, compliance, maint, critical_period
    ], dtype=float)
    return x

def gen_sample(rng: random.Random) -> Tuple[GeneralSignals, float]:
    gal = rng.choices(["none", "low", "medium", "high", "critical"], weights=[1, 2, 3, 2, 1])[0]
    exp = rng.choices(["low", "medium", "high"], weights=[2, 5, 3])[0]
    anomaly = max(0.0, min(1.0, rng.random() ** 0.7))
    incidents = int(np.clip(int(rng.gauss(3, 2)), 0, 20))
    patch_p50 = int(np.clip(int(rng.gauss(15, 8)), 0, 60))
    comp_debt = max(0.0, min(1.0, rng.random()))
    maint = rng.random() < 0.2
    crit_period = rng.random() < 0.25

    has_campaign = rng.random() < 0.6
    if has_campaign:
        sev = rng.choices(["low", "medium", "high"], weights=[2, 5, 3])[0]
        campaigns = [{"name": "synthetic-camp", "severity": sev, "geo": "global", "target_type": "api"}]
    else:
        campaigns = []

    s = GeneralSignals(
        global_alert_level=gal,
        current_campaigns=campaigns,
        anomaly_index_global=float(anomaly),
        incident_rate_7d=incidents,
        patch_delay_days_p50=patch_p50,
        exposure_level=exp,
        maintenance_window=maint,
        compliance_debt_score=comp_debt,
        business_critical_period=crit_period,
        geo_region="global"
    )

    w = {"base": 0.16, "intel": 0.12, "camp": 0.12, "inc": 0.14,
         "anom": 0.16, "patch": 0.10, "exp": 0.10, "comp": 0.10}
    base = SEVERITY_WEIGHTS[gal]
    intel = 0.5
    camp = campaigns_severity_max(campaigns)
    inc = min(1.0, incidents / 12.0)
    anom = anomaly
    patch = min(1.0, patch_p50 / 30.0)
    expv = EXPOSURE_WEIGHTS[exp]
    comp = comp_debt

    score = (
        w["base"] * base + w["intel"] * intel + w["camp"] * camp + w["inc"] * inc +
        w["anom"] * anom + w["patch"] * patch + w["exp"] * expv + w["comp"] * comp
    )
    multiplier = 1.0 + (0.05 if maint else 0.0) + (0.08 if crit_period else 0.0)
    score = float(np.clip(score * multiplier + rng.gauss(0, 0.03), 0.0, 1.0))
    return s, score

def gen_dataset(n: int, seed: int):
    rng = random.Random(seed)
    X_list, y_list = [], []
    for _ in range(n):
        s, y = gen_sample(rng)
        x = to_features(s)
        X_list.append(x)
        y_list.append(y)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    return X, y

class RiskModelService:
    def __init__(self):
        self.model: Optional[GradientBoostingRegressor] = None
        self.model_version: str = "untrained"
        self.cfg = ConfigRepository()
        self._lock = threading.Lock()

    def train(self, req: TrainRequest) -> TrainResponse:
        X, y = gen_dataset(req.n, req.seed if req.seed is not None else 42)

        model = GradientBoostingRegressor(
            random_state=req.seed or 42,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.06
        )
        model.fit(X, y)

        idx = int(0.8 * len(X))
        y_pred = model.predict(X[idx:])
        r2 = float(r2_score(y[idx:], y_pred))
        mae = float(mean_absolute_error(y[idx:], y_pred))

        with self._lock:
            self.model = model
            self.model_version = f"risk-ml-gbr-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        return TrainResponse(
            model_version=self.model_version,
            metrics={"r2": round(r2, 3), "mae": round(mae, 3)},
            samples=len(X)
        )

    def assess(self, req: AssessRequest) -> RiskContext:
        s = req.signals
        x = to_features(s).reshape(1, -1)

        with self._lock:
            model = self.model
            model_version = self.model_version

        if model is not None:
            pred = float(np.clip(model.predict(x)[0], 0.0, 1.0))
        else:
            pred = float(np.clip(
                0.15 * SEVERITY_WEIGHTS.get(s.global_alert_level, 0.5) +
                0.12 * 0.5 +
                0.12 * campaigns_severity_max(s.current_campaigns) +
                0.14 * min(1.0, s.incident_rate_7d / 12.0) +
                0.16 * s.anomaly_index_global +
                0.10 * min(1.0, s.patch_delay_days_p50 / 30.0) +
                0.10 * EXPOSURE_WEIGHTS.get(s.exposure_level, 0.5) +
                0.10 * s.compliance_debt_score, 0.0, 1.0
            ))
            if s.maintenance_window:
                pred *= 1.05
            if s.business_critical_period:
                pred *= 1.08
            pred = float(np.clip(pred, 0.0, 1.0))
            model_version = "risk-general-fallback-0.1.0"

        level = self._map_level(pred)
        policies = self.cfg.get_policy_overrides(level)

        threat_intel = {
            "global_severity": "medium",
            "campaigns": s.current_campaigns or [],
            "confidence": "medium"
        }

        return RiskContext(
            score=round(pred, 3),
            level=level,
            anomaly_score=round(s.anomaly_index_global, 3),
            threat_intel=threat_intel,
            recent_incidents=int(s.incident_rate_7d),
            policy_overrides=policies,
            timestamp=datetime.utcnow(),
            model_version=model_version
        )

    def _map_level(self, score: float) -> str:
        thr = self.cfg.get_policy_thresholds()
        if score >= thr["critical"]:
            return "critical"
        if score >= thr["high"]:
            return "high"
        if score >= thr["medium"]:
            return "medium"
        if score >= thr["low"]:
            return "low"
        return "very_low"