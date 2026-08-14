from __future__ import annotations

from pathlib import Path
import gc

import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger
from src.dataset_generation.core.user_generator import UserGenerator
from src.dataset_generation.core.event_generator import EventGenerator

# ✅ TODAS as importações de features corrigidas
from src.dataset_generation.features.event_identification import add_event_identification_features
from src.dataset_generation.features.user_account_features import add_user_account_features
from src.dataset_generation.features.temporal_features import add_temporal_behavioral_features
from src.dataset_generation.features.transaction_features import add_transaction_features
from src.dataset_generation.features.location_network_features import add_location_network_features
from src.dataset_generation.features.device_features import add_device_environment_features
from src.dataset_generation.features.security_features import add_security_tech_features
from src.dataset_generation.features.fraud_history_features import add_fraud_history_features
from src.dataset_generation.features.text_features import add_text_message_features
from src.dataset_generation.features.llm_features import add_llm_features

logger = get_logger("orchestrator")


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Otimiza uso de memória convertendo tipos de dados."""
    logger.info("Optimizing memory usage...")

    initial_memory = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # Converter object para category se tiver poucos valores únicos
        if col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

        # Converter float64 para float32
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')

        # Converter int64 para int32 se possível
        elif col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > -2147483648 and c_max < 2147483647:
                df[col] = df[col].astype('int32')

    final_memory = df.memory_usage(deep=True).sum() / 1024 ** 2
    reduction = (1 - final_memory / initial_memory) * 100

    logger.info(f"Memory reduced from {initial_memory:.1f} MB to {final_memory:.1f} MB ({reduction:.1f}% reduction)")

    return df


class DatasetOrchestrator:
    """Orchestrates the entire dataset generation pipeline."""

    def __init__(self, config_path: str = "dataset_config.yaml", output_dir: str = "output"):
        self.config = default_config_loader.load(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 80)
        logger.info("🚀 Starting dataset generation pipeline")
        logger.info("=" * 80)

        # 1. Generate users
        logger.info("\n[1/11] Generating users...")
        user_gen = UserGenerator()
        users_df = user_gen.generate_users()
        logger.info(f"✅ Generated {len(users_df):,} users")

        # 2. Generate raw events
        logger.info("\n[2/11] Generating raw events...")
        event_gen = EventGenerator(users_df)
        df = event_gen.generate_events()
        logger.info(f"✅ Generated {len(df):,} raw events")

        del event_gen
        gc.collect()

        # 3. Add features progressively
        logger.info("\n[3/11] Adding event identification features...")
        df = add_event_identification_features(df)
        gc.collect()

        logger.info("\n[4/11] Adding user account features...")
        df = add_user_account_features(df, users_df)
        gc.collect()

        del users_df
        gc.collect()

        logger.info("\n[5/11] Adding temporal behavioral features...")
        df = add_temporal_behavioral_features(df)
        gc.collect()

        logger.info("\n[6/11] Adding transaction features...")
        df = add_transaction_features(df)
        gc.collect()

        logger.info("\n[7/11] Adding location network features...")
        df = add_location_network_features(df)
        gc.collect()

        logger.info("\n[8/11] Adding device environment features...")
        df = add_device_environment_features(df)
        gc.collect()

        logger.info("\n[9/11] Adding security tech features...")
        df = add_security_tech_features(df)
        gc.collect()

        logger.info("\n[10/11] Adding fraud history features...")
        df = add_fraud_history_features(df)
        gc.collect()

        logger.info("\n[11/11] Adding text message features...")
        df = add_text_message_features(df)
        gc.collect()

        # LLM features (opcional)
        llm_enabled = self.config.get("llm", {}).get("enabled", False)
        if llm_enabled:
            logger.info("\n[BONUS] Adding LLM features...")
            df = add_llm_features(df)
            gc.collect()
        else:
            logger.info("\n[SKIP] LLM features disabled in config")

        # 4. Optimize memory
        logger.info("\n[OPTIMIZE] Optimizing memory usage...")
        df = optimize_dataframe_memory(df)
        gc.collect()

        # 5. Save dataset
        logger.info("\n[SAVE] Saving dataset...")
        self._save_dataset(df)

        logger.info("\n" + "=" * 80)
        logger.info("✅ Dataset generation pipeline completed successfully!")
        logger.info("=" * 80)

    def _save_dataset(self, df: pd.DataFrame):
        """Save the final dataset and splits."""
        logger.info("Sorting by timestamp...")
        try:
            df = df.sort_values("timestamp_utc")
        except MemoryError:
            logger.warning("Not enough memory for full sort, saving unsorted")

        df.reset_index(drop=True, inplace=True)
        gc.collect()

        # Save full dataset
        full_path = self.output_dir / "dataset_full.csv"
        logger.info(f"Saving full dataset to {full_path}")

        chunk_size = 500_000
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            chunk.to_csv(full_path, mode=mode, header=header, index=False)
            logger.info(f"  Saved chunk {i // chunk_size + 1}/{(len(df) - 1) // chunk_size + 1}")

        gc.collect()

        # Create splits
        self._create_splits(df)

        # Save summary
        self._save_summary(df)

    def _create_splits(self, df: pd.DataFrame):
        """Create train/val/test splits."""
        logger.info("Creating train/val/test splits...")

        splits_config = self.config.get("splits", {})
        strategy = splits_config.get("strategy", "time_based")

        if strategy == "time_based":
            train_ratio = splits_config.get("train_ratio", 0.70)
            val_ratio = splits_config.get("val_ratio", 0.15)

            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

        else:  # random
            train_df = df.sample(frac=0.70, random_state=42)
            remaining = df.drop(train_df.index)
            val_df = remaining.sample(frac=0.5, random_state=42)
            test_df = remaining.drop(val_df.index)

        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }

        chunk_size = 500_000
        for split_name, split_df in splits.items():
            split_path = self.output_dir / f"dataset_{split_name}.csv"
            logger.info(f"Saving {split_name} split ({len(split_df):,} rows) to {split_path}")

            for i in range(0, len(split_df), chunk_size):
                chunk = split_df.iloc[i:i + chunk_size]
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                chunk.to_csv(split_path, mode=mode, header=header, index=False)

            gc.collect()

        logger.info("✅ Splits created successfully")

    def _save_summary(self, df: pd.DataFrame):
        """Save a summary report."""
        summary_path = self.output_dir / "dataset_summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("FRAUD DETECTION DATASET - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Events: {len(df):,}\n")
            f.write(f"Total Columns: {len(df.columns)}\n")
            f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB\n\n")

            splits_config = self.config.get("splits", {})
            train_ratio = splits_config.get("train_ratio", 0.70)
            val_ratio = splits_config.get("val_ratio", 0.15)
            test_ratio = splits_config.get("test_ratio", 0.15)

            f.write("Dataset Splits:\n")
            f.write(f"  full  : {len(df):>10,} rows\n")
            f.write(f"  train : {int(len(df) * train_ratio):>10,} rows\n")
            f.write(f"  val   : {int(len(df) * val_ratio):>10,} rows\n")
            f.write(f"  test  : {int(len(df) * test_ratio):>10,} rows\n\n")

            if "is_fraud" in df.columns:
                fraud_count = df["is_fraud"].sum()
                legit_count = len(df) - fraud_count
                f.write("Fraud Distribution:\n")
                f.write(f"  Fraudulent:  {fraud_count:>10,} ({fraud_count / len(df) * 100:>5.2f}%)\n")
                f.write(f"  Legitimate:  {legit_count:>10,} ({legit_count / len(df) * 100:>5.2f}%)\n\n")

            f.write("Column List:\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"  {i:>3}. {col}\n")

        logger.info(f"✅ Summary saved to {summary_path}")