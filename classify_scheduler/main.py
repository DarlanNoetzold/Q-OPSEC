import argparse
from datetime import datetime
from pathlib import Path

from configs import DefaultConfig
from data import load_dataset
from utils import seed_everything, normalize_labels, ensure_classes, print_summary_classes
from trainer import train_and_select_best
from evaluator import evaluate_holdout
from registry import ModelRegistry
from scheduler import schedule_periodic_training


def run_train_once(cfg: DefaultConfig):
    seed_everything(cfg.seed)
    df = load_dataset(cfg.data_path)

    # Normaliza rótulos
    df, dropped = normalize_labels(df, cfg.target_col)
    if dropped:
        print(f"[WARN] Dropped rows with invalid labels: {dropped}")

    # Falha se o CSV não tiver pelo menos 2 classes
    ensure_classes(df, cfg.target_col, cfg.allowed_classes)
    print_summary_classes(df, cfg.target_col)

    best = train_and_select_best(
        df=df,
        target_col=cfg.target_col,
        test_size=cfg.test_size,
        seed=cfg.seed,
        n_splits=cfg.cv_splits,
        scoring_primary="accuracy",
        scoring_secondary="f1_macro",
        max_models=cfg.max_models,
    )

    print(f"[INFO] Classes originais (LabelEncoder): {list(best.label_encoder.classes_)}")
    print(f"[INFO] Classes ordenadas por severidade: {best.ordered_classes}")
    print(f"[INFO] Feature columns (para API): {best.feature_columns}")
    print(f"[INFO] Mapeamento final: {dict(enumerate(best.ordered_classes))}")

    print(
        f"[INFO] Best model: {best.best_name} "
        f"(acc_cv={best.best_cv_metrics['accuracy']:.4f}, f1_macro_cv={best.best_cv_metrics['f1_macro']:.4f})"
    )

    holdout = evaluate_holdout(best.pipeline, best.X_test, best.y_test, cfg.allowed_classes)
    print(f"[HOLDOUT] accuracy={holdout['accuracy']:.4f}, f1_macro={holdout['f1_macro']:.4f}")

    # USAR FEATURE COLUMNS (não todas as colunas do CSV)
    required_columns = list(best.feature_columns)
    print(f"[INFO] Required columns for API: {required_columns}")

    registry = ModelRegistry(cfg.registry_dir)
    tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    save_path = registry.save(
        tag=tag,
        pipeline=best.pipeline,
        classes=best.ordered_classes,  # USAR CLASSES ORDENADAS, NÃO label_encoder.classes_
        cv_metrics=best.best_cv_metrics,
        holdout_metrics=holdout,
        model_name=best.best_name,
        meta={
            "seed": cfg.seed,
            "cv_splits": cfg.cv_splits,
            "test_size": cfg.test_size,
            "data_path": str(cfg.data_path),
            "target_col": cfg.target_col,
            "n_samples": len(df),
            "n_features": len(required_columns),
            "required_columns": required_columns,  # Só feature columns
        },
        candidates=best.candidates,
    )
    print(f"[REGISTRY] Saved best model '{best.best_name}' to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="classify_scheduler")
    parser.add_argument("--mode", choices=["train-once", "schedule"], default="train-once")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--registry", type=str, default="./model_registry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-models", type=int, default=20)
    parser.add_argument("--target-col", type=str, default="security_level_label")
    parser.add_argument("--cron", type=str, default="0 */6 * * *")
    args = parser.parse_args()

    cfg = DefaultConfig(
        data_path=Path(args.data),
        registry_dir=Path(args.registry),
        seed=args.seed,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        max_models=args.max_models,
        target_col=args.target_col,
    )

    if args.mode == "train-once":
        run_train_once(cfg)
    else:
        schedule_periodic_training(cron_expr=args.cron, cfg=cfg, job_fn=run_train_once)


if __name__ == "__main__":
    main()