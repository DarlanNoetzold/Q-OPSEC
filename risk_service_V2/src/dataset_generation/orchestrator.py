from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger
from src.dataset_generation.core.user_generator import UserGenerator
from src.dataset_generation.core.event_generator import EventGenerator
from src.dataset_generation.features.event_identification import add_event_identification_features
from src.dataset_generation.features.user_account_features import add_user_account_features
from src.dataset_generation.features.temporal_features import add_temporal_behavioral_features  # CORRIGIDO
from src.dataset_generation.features.device_environment_features import add_device_environment_features
from src.dataset_generation.features.security_tech_features import add_security_tech_features
from src.dataset_generation.features.fraud_history_features import add_fraud_history_features
from src.dataset_generation.features.text_features import add_text_features  # CORRIGIDO (assumindo que você renomeou)
from src.dataset_generation.features.llm_features import add_llm_features
from src.dataset_generation.labels.labeling import apply_labels


logger = get_logger("dataset_orchestrator")


@dataclass
class DatasetOrchestrator:
    """Main orchestrator for synthetic dataset generation."""

    output_dir: Path
    dataset_config_path: str = "dataset_config.yaml"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = default_config_loader.load(self.dataset_config_path)

    def run(self) -> None:
        logger.info("Starting dataset generation pipeline")

        # 1) Generate users
        user_gen = UserGenerator(self.config)
        users = user_gen.generate_users()
        logger.info("Generated {n} users", n=len(users))

        # 2) Generate events
        event_gen = EventGenerator(self.config)
        events = event_gen.generate_events(users)
        logger.info("Generated {n} raw events", n=len(events))

        # 3) Feature engineering pipeline
        df = events

        # 3.1 Event identification
        df = add_event_identification_features(df)

        # 3.2 User/account features (join with user table)
        df = add_user_account_features(df, users)

        # 3.3 Temporal / behavioral features
        df = add_temporal_behavioral_features(df)  # CORRIGIDO

        # 3.4 Device / environment
        df = add_device_environment_features(df)

        # 3.5 Security technologies & versions
        df = add_security_tech_features(df)

        # 3.6 Fraud / historical risk
        df = add_fraud_history_features(df)

        # 3.7 Text / content features
        df = add_text_features(df)  # CORRIGIDO (se você renomeou a função)

        # 3.8 LLM-derived features
        df = add_llm_features(df)

        # 4) Labels / ground truth
        df = apply_labels(df)

        # 5) Split and save
        self._save_dataset(df)

        logger.info("Dataset generation pipeline completed")

    # ------------------------------------------------------------------
    # Saving / splitting
    # ------------------------------------------------------------------

    def _save_dataset(self, df: pd.DataFrame) -> None:
        export_cfg = self.config.get("export", {})
        fmt = export_cfg.get("format", "parquet")
        splits = export_cfg.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})

        # Sort by time for time-based split
        if "timestamp_utc" in df.columns:
            df = df.sort_values("timestamp_utc").reset_index(drop=True)

        n = len(df)
        train_size = int(n * splits.get("train", 0.7))
        val_size = int(n * splits.get("val", 0.15))
        test_size = n - train_size - val_size

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]

        logger.info(
            "Dataset split sizes - train: {t}, val: {v}, test: {s}",
            t=len(train_df),
            v=len(val_df),
            s=len(test_df),
        )

        # Save
        if fmt == "csv":
            train_df.to_csv(self.output_dir / "dataset_train.csv", index=False)
            val_df.to_csv(self.output_dir / "dataset_val.csv", index=False)
            test_df.to_csv(self.output_dir / "dataset_test.csv", index=False)
            df.to_csv(self.output_dir / "dataset_full.csv", index=False)
        else:
            train_df.to_parquet(self.output_dir / "dataset_train.parquet", index=False)
            val_df.to_parquet(self.output_dir / "dataset_val.parquet", index=False)
            test_df.to_parquet(self.output_dir / "dataset_test.parquet", index=False)
            df.to_parquet(self.output_dir / "dataset_full.parquet", index=False)

        logger.info("Datasets saved to {path}", path=str(self.output_dir))