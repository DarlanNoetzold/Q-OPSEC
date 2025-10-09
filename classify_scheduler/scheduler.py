from typing import Callable
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time
import os
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

METRICS_ROOT = Path(os.getcwd()) / "models" / "metrics"
METRICS_ROOT.mkdir(parents=True, exist_ok=True)


def save_training_metrics(train_result, session_id: str):
    """
    Salva métricas e gráficos do treinamento
    train_result: TrainResult do trainer.py
    """
    session_path = METRICS_ROOT / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    candidates = [c for c in train_result.candidates if "error" not in c]
    if not candidates:
        print("[WARN] No successful candidates to save metrics")
        return

    names = [c["name"] for c in candidates]
    accs = [c["cv_accuracy"] for c in candidates]
    f1s = [c["cv_f1_macro"] for c in candidates]

    # 1. Accuracy comparison
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if v == max(accs) else '#3498db' for v in accs]
    plt.barh(names, accs, color=colors)
    plt.xlabel('CV Accuracy')
    plt.title('Model Comparison - Cross-Validation Accuracy')
    plt.xlim(0, 1.0)
    for i, v in enumerate(accs):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(session_path / "all_models_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. F1-Score comparison
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if v == max(f1s) else '#e74c3c' for v in f1s]
    plt.barh(names, f1s, color=colors)
    plt.xlabel('CV F1-Score (Macro)')
    plt.title('Model Comparison - Cross-Validation F1-Score')
    plt.xlim(0, 1.0)
    for i, v in enumerate(f1s):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(session_path / "all_models_f1score.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Accuracy vs F1 scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(accs, f1s, s=100, alpha=0.6, c=range(len(names)), cmap='viridis')
    for i, name in enumerate(names):
        plt.annotate(name, (accs[i], f1s[i]), fontsize=8, alpha=0.7)
    plt.xlabel('CV Accuracy')
    plt.ylabel('CV F1-Score (Macro)')
    plt.title('Accuracy vs F1-Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(session_path / "accuracy_vs_f1.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Confusion matrix (holdout)
    y_pred = train_result.pipeline.predict(train_result.X_test)
    cm = confusion_matrix(train_result.y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_result.ordered_classes,
                yticklabels=train_result.ordered_classes)
    plt.title('Confusion Matrix - Best Model (Holdout)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(session_path / "best_model_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Top 10 ranking
    top10 = sorted(candidates, key=lambda x: x["cv_accuracy"], reverse=True)[:10]
    if top10:
        top_names = [c["name"] for c in top10]
        top_acc = [c["cv_accuracy"] for c in top10]
        top_f1 = [c["cv_f1_macro"] for c in top10]

        x = range(len(top_names))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar([i - width / 2 for i in x], top_acc, width, label='Accuracy', color='#3498db')
        plt.bar([i + width / 2 for i in x], top_f1, width, label='F1-Score', color='#e74c3c')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Top 10 Models Ranking')
        plt.xticks(list(x), top_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(session_path / "top10_models_ranking.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 6. Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "best_model": {
            "name": train_result.best_name,
            "cv_metrics": train_result.best_cv_metrics,
        },
        "statistics": {
            "avg_accuracy": float(np.mean(accs)),
            "avg_f1_score": float(np.mean(f1s)),
            "best_accuracy": float(max(accs)),
            "best_f1_score": float(max(f1s)),
            "worst_accuracy": float(min(accs)),
            "worst_f1_score": float(min(f1s)),
        },
        "dataset_info": {
            "test_samples": int(len(train_result.y_test)),
            "features": train_result.feature_columns,
            "classes": train_result.ordered_classes,
        },
        "all_models": candidates,
    }

    with open(session_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metricas salvas em: {session_path}")


def schedule_periodic_training(cron_expr: str, cfg, job_fn: Callable):
    sched = BackgroundScheduler()
    trigger = CronTrigger.from_crontab(cron_expr)
    sched.add_job(job_fn, trigger=trigger, args=[cfg], id="periodic_training", replace_existing=True)
    sched.start()
    print(f"[SCHEDULER] Started with CRON '{cron_expr}'. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("[SCHEDULER] Stopping...")
        sched.shutdown()