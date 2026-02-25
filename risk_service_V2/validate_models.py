import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    brier_score_loss, matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS_ROOT   = Path("output/models")
EVAL_ROOT     = Path("output/evaluation")
DATA_ROOT     = Path("output")
OUTPUT_DIR    = Path("output/validation")
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

TARGET_COL    = "is_fraud"
EXCLUDE_COLS  = [
    "event_id", "user_id", "account_id", "is_fraud", "fraud_type",
    "timestamp_utc", "timestamp_local", "account_creation_date",
    "message_text", "llm_risk_reasoning", "transaction_description",
    "ip_address", "device_id",
]

MODEL_DISPLAY = {
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "catboost":            "CatBoost",
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
    "pytorch_mlp":         "MLP (PyTorch)",
}

PALETTE = {
    "xgboost":             "#2196F3",
    "lightgbm":            "#4CAF50",
    "catboost":            "#FF9800",
    "random_forest":       "#9C27B0",
    "logistic_regression": "#F44336",
    "pytorch_mlp":         "#00BCD4",
}

STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}
plt.rcParams.update(STYLE)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_latest_version(models_root: Path) -> Path:
    dirs = sorted([d for d in models_root.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No model versions found in {models_root}")
    return dirs[-1]


def load_feature_names(version_dir: Path):
    p = version_dir / "feature_names.json"
    if not p.exists():
        return []
    with open(p) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for k in ("all_features", "features", "feature_list"):
            if k in raw and isinstance(raw[k], list):
                return raw[k]
    return list(raw.keys()) if isinstance(raw, dict) else []


def load_artifacts(version_dir: Path):
    encoders = {}
    enc_path = version_dir / "label_encoders.pkl"
    if enc_path.exists():
        encoders = joblib.load(enc_path)

    scaler = None
    sc_path = version_dir / "scaler.pkl"
    if sc_path.exists():
        scaler = joblib.load(sc_path)

    models, metadata = {}, {}
    for p in version_dir.glob("*_model.pkl"):
        name = p.stem.replace("_model", "")
        try:
            models[name] = joblib.load(p)
        except Exception as e:
            print(f"  [WARN] Could not load {p.name}: {e}")

    for p in version_dir.glob("*_metadata.json"):
        name = p.stem.replace("_metadata", "")
        with open(p) as f:
            metadata[name] = json.load(f)

    return models, metadata, encoders, scaler


def load_dataset(split: str = "test") -> pd.DataFrame:
    for ext in ("parquet", "csv"):
        p = DATA_ROOT / f"dataset_{split}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"Dataset split '{split}' not found in {DATA_ROOT}")


def preprocess(df: pd.DataFrame, feature_names, encoders, scaler, model_name: str) -> pd.DataFrame:
    df = df.copy().replace("", np.nan)

    # apply label encoders
    for col, enc in encoders.items():
        if col not in df.columns:
            continue
        vals = df[col].astype(str).fillna("__NA__").values
        try:
            df[col] = enc.transform(vals)
        except Exception:
            mapping = {c: i for i, c in enumerate(enc.classes_)}
            df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)

    # lightgbm: keep categoricals as pd.Categorical
    if model_name == "lightgbm":
        for col in encoders.keys():
            if col in df.columns:
                cats = list(encoders[col].classes_)
                fill = "missing" if "missing" in cats else (cats[0] if cats else None)
                df[col] = pd.Categorical(df[col].astype(object).fillna(fill), categories=cats)
        for col in df.columns:
            if col not in encoders:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ensure all expected features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # pytorch_mlp: force float64
    if model_name == "pytorch_mlp":
        df = df.astype(np.float64)
        return df

    # apply scaler (numeric only, skip lightgbm categoricals)
    if scaler is not None and model_name != "logistic_regression":
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df[num_cols] = scaler.transform(df[num_cols])
        except Exception:
            pass

    return df


def get_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim > 1 else p.ravel()
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    return model.predict(X).astype(float)


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity":       float(specificity),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "pr_auc":            float(average_precision_score(y_true, y_prob)),
        "log_loss":          float(log_loss(y_true, y_prob)),
        "brier_score":       float(brier_score_loss(y_true, y_prob)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "npv":               float(npv),
        "threshold":         float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n=1000, ci=0.95, threshold=0.5):
    rng = np.random.default_rng(RANDOM_SEED)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx], threshold))
        except Exception:
            pass
    scores = np.array(scores)
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lo), float(hi)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_roc_curves(results: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        auc = r["metrics"]["roc_auc"]
        label = f"{MODEL_DISPLAY.get(name, name)} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, color=PALETTE.get(name, None), lw=1.8, label=label)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png")
    plt.close(fig)


def plot_pr_curves(results: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        ap = r["metrics"]["pr_auc"]
        label = f"{MODEL_DISPLAY.get(name, name)} (AP={ap:.3f})"
        ax.plot(rec, prec, color=PALETTE.get(name, None), lw=1.8, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png")
    plt.close(fig)


def plot_confusion_matrices(results: dict, out_dir: Path):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).ravel()
    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]
        cm = confusion_matrix(r["y_true"], (r["y_prob"] >= r["metrics"]["threshold"]).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
                    cbar=False, linewidths=0.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(MODEL_DISPLAY.get(name, name), fontsize=10, pad=4)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png")
    plt.close(fig)


def plot_metrics_comparison(results: dict, out_dir: Path):
    metrics_to_plot = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "mcc", "brier_score"]
    labels = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy", "MCC", "Brier Score"]

    rows = []
    for name, r in results.items():
        for m, lbl in zip(metrics_to_plot, labels):
            rows.append({
                "model": MODEL_DISPLAY.get(name, name),
                "metric": lbl,
                "value": r["metrics"][m],
            })
    df_plot = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics_to_plot))
    width = 0.13
    model_names = list(results.keys())
    for i, name in enumerate(model_names):
        vals = [results[name]["metrics"][m] for m in metrics_to_plot]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_DISPLAY.get(name, name),
                      color=PALETTE.get(name, None), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_comparison.png")
    plt.close(fig)


def plot_calibration_curves(results: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    for name, r in results.items():
        frac_pos, mean_pred = calibration_curve(r["y_true"], r["y_prob"], n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", color=PALETTE.get(name, None),
                lw=1.5, ms=4, label=MODEL_DISPLAY.get(name, name))
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curves.png")
    plt.close(fig)


def plot_threshold_analysis(results: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]
        thresholds = np.linspace(0.05, 0.95, 91)
        f1s, precs, recs = [], [], []
        for t in thresholds:
            yp = (r["y_prob"] >= t).astype(int)
            f1s.append(f1_score(r["y_true"], yp, zero_division=0))
            precs.append(precision_score(r["y_true"], yp, zero_division=0))
            recs.append(recall_score(r["y_true"], yp, zero_division=0))
        ax.plot(thresholds, f1s,   lw=1.8, label="F1",        color="#2196F3")
        ax.plot(thresholds, precs, lw=1.8, label="Precision",  color="#4CAF50")
        ax.plot(thresholds, recs,  lw=1.8, label="Recall",     color="#F44336")
        opt_t = r["metrics"]["threshold"]
        ax.axvline(opt_t, color="gray", lw=1, ls="--", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(MODEL_DISPLAY.get(name, name), fontsize=9, pad=3)
        ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_analysis.png")
    plt.close(fig)


def plot_score_distributions(results: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]
        y_true = r["y_true"]
        y_prob = r["y_prob"]
        ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="#2196F3",
                density=True, label="Legit")
        ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="#F44336",
                density=True, label="Fraud")
        ax.axvline(r["metrics"]["threshold"], color="gray", lw=1.2, ls="--")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.set_title(MODEL_DISPLAY.get(name, name), fontsize=9, pad=3)
        ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png")
    plt.close(fig)


def plot_feature_importance(results: dict, feature_names: list, out_dir: Path, top_n: int = 25):
    importances = {}
    for name, r in results.items():
        model = r.get("model")
        if model is None:
            continue
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        if imp is not None and len(imp) == len(feature_names):
            importances[name] = imp

    if not importances:
        return

    # individual plots
    for name, imp in importances.items():
        idx = np.argsort(imp)[-top_n:]
        fig, ax = plt.subplots(figsize=(7, top_n * 0.28 + 1))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(idx)))
        ax.barh([feature_names[i] for i in idx], imp[idx], color=colors)
        ax.set_xlabel("Importance")
        ax.set_title(MODEL_DISPLAY.get(name, name), fontsize=10, pad=4)
        fig.tight_layout()
        fig.savefig(out_dir / f"feature_importance_{name}.png")
        plt.close(fig)

    # aggregated heatmap (top_n features by mean importance)
    df_imp = pd.DataFrame(importances, index=feature_names)
    df_imp["mean"] = df_imp.mean(axis=1)
    top_feats = df_imp.nlargest(top_n, "mean").drop(columns="mean")
    if top_feats.empty:
        return
    # normalize per model
    top_feats_norm = top_feats.div(top_feats.max(axis=0), axis=1).fillna(0)
    top_feats_norm.columns = [MODEL_DISPLAY.get(c, c) for c in top_feats_norm.columns]

    fig, ax = plt.subplots(figsize=(len(top_feats_norm.columns) * 1.4 + 2, top_n * 0.3 + 1))
    sns.heatmap(top_feats_norm, ax=ax, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "Normalized importance"}, annot=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance_heatmap.png")
    plt.close(fig)


def plot_bootstrap_ci(ci_results: dict, metric: str, out_dir: Path):
    names, means, los, his = [], [], [], []
    for name, r in ci_results.items():
        if metric not in r:
            continue
        m, lo, hi = r[metric]
        names.append(MODEL_DISPLAY.get(name, name))
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=[los, his], color=[PALETTE.get(n.lower().replace(" ", "_"), "#888")
                                               for n in names],
            alpha=0.8, capsize=4, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric.upper().replace("_", "-"))
    ax.set_xlim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.05))
    fig.tight_layout()
    fig.savefig(out_dir / f"bootstrap_ci_{metric}.png")
    plt.close(fig)


def plot_dataset_stats(df_test: pd.DataFrame, out_dir: Path):
    """Feature distribution overview for the test set."""
    numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL][:16]

    if not numeric_cols:
        return

    cols = 4
    rows = (len(numeric_cols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    axes = np.array(axes).ravel()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label, color in [(0, "#2196F3"), (1, "#F44336")]:
            subset = df_test.loc[df_test[TARGET_COL] == label, col].dropna()
            ax.hist(subset, bins=30, alpha=0.55, color=color,
                    density=True, label=("Legit" if label == 0 else "Fraud"))
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png")
    plt.close(fig)


def plot_correlation_matrix(df_test: pd.DataFrame, feature_names: list, out_dir: Path, top_n: int = 30):
    num_feats = [f for f in feature_names
                 if f in df_test.columns and pd.api.types.is_numeric_dtype(df_test[f])][:top_n]
    if len(num_feats) < 2:
        return
    corr = df_test[num_feats].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(top_n * 0.45 + 2, top_n * 0.4 + 2))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
                linewidths=0.2, annot=False, cbar_kws={"shrink": 0.7})
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation_matrix.png")
    plt.close(fig)


def plot_wilcoxon_heatmap(results: dict, out_dir: Path):
    """Pairwise Wilcoxon signed-rank test on predicted probabilities."""
    names = list(results.keys())
    n = len(names)
    pval_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                _, p = wilcoxon(results[names[i]]["y_prob"],
                                results[names[j]]["y_prob"])
                pval_matrix[i, j] = p
            except Exception:
                pval_matrix[i, j] = 1.0

    display_names = [MODEL_DISPLAY.get(n, n) for n in names]
    df_p = pd.DataFrame(-np.log10(pval_matrix + 1e-300), index=display_names, columns=display_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_p, ax=ax, cmap="Blues", annot=True, fmt=".1f",
                linewidths=0.4, cbar_kws={"label": "-log10(p-value)"})
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "wilcoxon_pvalue_heatmap.png")
    plt.close(fig)


def plot_det_curves(results: dict, out_dir: Path):
    """Detection Error Tradeoff (DET) curve."""
    from scipy.stats import norm as scipy_norm

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, fnr, _ = roc_curve(r["y_true"], r["y_prob"])
        fnr = 1 - fpr[::-1]
        fpr_det = fpr[::-1]
        # transform to normal deviate
        fpr_t = scipy_norm.ppf(np.clip(fpr_det, 1e-4, 1 - 1e-4))
        fnr_t = scipy_norm.ppf(np.clip(fnr, 1e-4, 1 - 1e-4))
        ax.plot(fpr_t, fnr_t, color=PALETTE.get(name, None), lw=1.8,
                label=MODEL_DISPLAY.get(name, name))

    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
    tick_labels = [f"{t*100:.1f}%" for t in ticks]
    tick_pos = [scipy_norm.ppf(t) for t in ticks]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "det_curves.png")
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load version
    if args.version:
        version_dir = MODELS_ROOT / args.version
    else:
        version_dir = get_latest_version(MODELS_ROOT)
    print(f"\n📦 Model version: {version_dir.name}")

    feature_names = load_feature_names(version_dir)
    models, metadata, encoders, scaler = load_artifacts(version_dir)
    print(f"   Features : {len(feature_names)}")
    print(f"   Models   : {list(models.keys())}")

    # ── load test set
    df_test = load_dataset("test")
    print(f"\n📊 Test set: {df_test.shape[0]:,} rows × {df_test.shape[1]} cols")
    print(f"   Fraud rate: {df_test[TARGET_COL].mean()*100:.2f}%")

    y_true = df_test[TARGET_COL].values.astype(int)

    # ── dataset plots
    print("\n🖼  Generating dataset plots...")
    plot_dataset_stats(df_test, out_dir)
    plot_correlation_matrix(df_test, feature_names, out_dir)

    # ── per-model evaluation
    results = {}
    print("\n🔍 Evaluating models...")
    for name, model in models.items():
        print(f"   → {MODEL_DISPLAY.get(name, name)}")
        try:
            X = preprocess(df_test.drop(columns=[c for c in EXCLUDE_COLS if c in df_test.columns],
                                        errors="ignore"),
                           feature_names, encoders, scaler, name)
            y_prob = get_probabilities(model, X)
            threshold = float(metadata.get(name, {}).get("optimal_threshold", 0.5))
            metrics = compute_metrics(y_true, y_prob, threshold)
            results[name] = {
                "y_true": y_true,
                "y_prob": y_prob,
                "metrics": metrics,
                "model": model,
            }
            print(f"      AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")
        except Exception as e:
            print(f"      [ERROR] {e}")

    if not results:
        print("❌ No models evaluated successfully.")
        sys.exit(1)

    # ── bootstrap CI
    print("\n📐 Computing bootstrap confidence intervals (n=1000)...")
    ci_results = {}
    for name, r in results.items():
        ci_results[name] = {}
        for metric, fn in [
            ("roc_auc",  lambda yt, yp, t: roc_auc_score(yt, yp)),
            ("pr_auc",   lambda yt, yp, t: average_precision_score(yt, yp)),
            ("f1",       lambda yt, yp, t: f1_score(yt, (yp >= t).astype(int), zero_division=0)),
        ]:
            mean, lo, hi = bootstrap_ci(r["y_true"], r["y_prob"], fn, threshold=r["metrics"]["threshold"])
            ci_results[name][metric] = (mean, lo, hi)
        print(f"   {MODEL_DISPLAY.get(name, name)}: "
              f"AUC={ci_results[name]['roc_auc'][0]:.4f} "
              f"[{ci_results[name]['roc_auc'][1]:.4f}, {ci_results[name]['roc_auc'][2]:.4f}]")

    # ── plots
    print("\n🖼  Generating model plots...")
    plot_roc_curves(results, out_dir)
    plot_pr_curves(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_metrics_comparison(results, out_dir)
    plot_calibration_curves(results, out_dir)
    plot_threshold_analysis(results, out_dir)
    plot_score_distributions(results, out_dir)
    plot_feature_importance(results, feature_names, out_dir)
    plot_bootstrap_ci(ci_results, "roc_auc", out_dir)
    plot_bootstrap_ci(ci_results, "f1", out_dir)
    plot_wilcoxon_heatmap(results, out_dir)
    plot_det_curves(results, out_dir)

    # ── summary table
    print("\n📋 Building summary table...")
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        ci_auc = ci_results.get(name, {}).get("roc_auc", (None, None, None))
        ci_f1  = ci_results.get(name, {}).get("f1",      (None, None, None))
        rows.append({
            "Model":          MODEL_DISPLAY.get(name, name),
            "Accuracy":       round(m["accuracy"],    4),
            "Precision":      round(m["precision"],   4),
            "Recall":         round(m["recall"],      4),
            "Specificity":    round(m["specificity"], 4),
            "F1":             round(m["f1"],          4),
            "ROC-AUC":        round(m["roc_auc"],     4),
            "PR-AUC":         round(m["pr_auc"],      4),
            "MCC":            round(m["mcc"],         4),
            "Cohen Kappa":    round(m["cohen_kappa"], 4),
            "Brier Score":    round(m["brier_score"], 4),
            "Log Loss":       round(m["log_loss"],    4),
            "NPV":            round(m["npv"],         4),
            "Threshold":      round(m["threshold"],   4),
            "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
            "AUC CI 95% Lo":  round(ci_auc[1], 4) if ci_auc[1] else None,
            "AUC CI 95% Hi":  round(ci_auc[2], 4) if ci_auc[2] else None,
            "F1 CI 95% Lo":   round(ci_f1[1],  4) if ci_f1[1]  else None,
            "F1 CI 95% Hi":   round(ci_f1[2],  4) if ci_f1[2]  else None,
        })

    df_summary = pd.DataFrame(rows).set_index("Model")
    df_summary.to_csv(out_dir / "metrics_summary.csv")
    print(df_summary[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC"]].to_string())

    # ── feature importance table
    print("\n📋 Building feature importance table...")
    fi_rows = []
    for name, r in results.items():
        model = r.get("model")
        if model is None:
            continue
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        if imp is not None and len(imp) == len(feature_names):
            for feat, val in zip(feature_names, imp):
                fi_rows.append({"model": MODEL_DISPLAY.get(name, name), "feature": feat, "importance": float(val)})

    if fi_rows:
        df_fi = pd.DataFrame(fi_rows)
        df_fi.to_csv(out_dir / "feature_importance_all.csv", index=False)

        # pivot: mean importance across models
        df_fi_pivot = df_fi.pivot_table(index="feature", columns="model", values="importance", aggfunc="mean")
        df_fi_pivot["mean"] = df_fi_pivot.mean(axis=1)
        df_fi_pivot = df_fi_pivot.sort_values("mean", ascending=False)
        df_fi_pivot.to_csv(out_dir / "feature_importance_pivot.csv")

    # ── JSON report
    report = {
        "version":    version_dir.name,
        "timestamp":  datetime.now().isoformat(),
        "dataset": {
            "test_samples":  int(len(y_true)),
            "fraud_samples": int(y_true.sum()),
            "legit_samples": int((y_true == 0).sum()),
            "fraud_rate":    float(y_true.mean()),
            "n_features":    len(feature_names),
        },
        "models": {
            name: {
                "metrics": r["metrics"],
                "bootstrap_ci": {
                    k: {"mean": v[0], "ci_lo": v[1], "ci_hi": v[2]}
                    for k, v in ci_results.get(name, {}).items()
                },
            }
            for name, r in results.items()
        },
    }
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Validation complete. Outputs saved to: {out_dir.resolve()}")
    print("   Files generated:")
    for p in sorted(out_dir.iterdir()):
        print(f"     {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model validation for academic paper")
    parser.add_argument("--output_dir", default="output/validation",
                        help="Directory to save all outputs")
    parser.add_argument("--version", default=None,
                        help="Model version to evaluate (default: latest)")
    args = parser.parse_args()
    main(args)