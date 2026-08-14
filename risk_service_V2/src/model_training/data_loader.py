"""
Data Loader - Load and prepare datasets for model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

from src.common.logger import logger


class DataLoader:
    """Load and prepare datasets for training."""

    def __init__(self, config: dict):
        """
        Initialize DataLoader.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.input_dir = Path(self.data_config.get("input_dir", "output"))
        self.target_column = self.data_config.get("target_column", "is_fraud")
        self.exclude_columns = self.data_config.get("exclude_columns", [])

        # Feature types
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_columns: List[str] = []

        logger.info("DataLoader initialized")

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test datasets.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 80)
        logger.info("LOADING DATASETS")
        logger.info("=" * 80)

        train_file = self.data_config.get("train_file", "dataset_train.parquet")
        val_file = self.data_config.get("val_file", "dataset_val.parquet")
        test_file = self.data_config.get("test_file", "dataset_test.parquet")

        train_df = self._load_file(train_file, "train")
        val_df = self._load_file(val_file, "validation")
        test_df = self._load_file(test_file, "test")

        logger.info(f"\n✅ All datasets loaded successfully")
        logger.info(f"   Train: {len(train_df):,} rows")
        logger.info(f"   Val:   {len(val_df):,} rows")
        logger.info(f"   Test:  {len(test_df):,} rows")

        return train_df, val_df, test_df

    def _load_file(self, filename: str, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset file (parquet or csv)."""
        filepath = self.input_dir / filename

        # Try parquet first
        if filepath.suffix == ".parquet" and filepath.exists():
            logger.info(f"📂 Loading {dataset_name}: {filename}")
            df = pd.read_parquet(filepath)
            logger.info(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df

        # Try csv
        csv_path = filepath.with_suffix(".csv")
        if csv_path.exists():
            logger.info(f"📂 Loading {dataset_name}: {csv_path.name}")
            df = pd.read_csv(csv_path)
            logger.info(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df

        raise FileNotFoundError(f"Dataset file not found: {filepath} or {csv_path}")

    def prepare_features(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING FEATURES")
        logger.info("=" * 80)

        # Check target column exists
        if self.target_column not in train_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Identify feature columns
        self.feature_columns = [
            col for col in train_df.columns
            if col not in self.exclude_columns and col != self.target_column
        ]

        logger.info(f"\n📋 Feature Selection:")
        logger.info(f"   Total columns:    {len(train_df.columns)}")
        logger.info(f"   Excluded columns: {len(self.exclude_columns)}")
        logger.info(f"   Feature columns:  {len(self.feature_columns)}")

        # Identify feature types
        self._identify_feature_types(train_df)

        # Extract features and target
        X_train = train_df[self.feature_columns].copy()
        y_train = train_df[self.target_column].copy()

        X_val = val_df[self.feature_columns].copy()
        y_val = val_df[self.target_column].copy()

        X_test = test_df[self.feature_columns].copy()
        y_test = test_df[self.target_column].copy()

        # 👇 NOVO: Limpar NaN do target ANTES de processar features
        logger.info("\n🎯 Cleaning target variable:")
        logger.info(f"   Train y NaN: {y_train.isnull().sum()}")
        logger.info(f"   Val y NaN:   {y_val.isnull().sum()}")
        logger.info(f"   Test y NaN:  {y_test.isnull().sum()}")

        # Remove rows where target is NaN
        if y_train.isnull().any():
            mask = ~y_train.isnull()
            X_train = X_train[mask]
            y_train = y_train[mask]
            logger.warning(f"   ⚠️  Removed {(~mask).sum()} rows with NaN target from train")

        if y_val.isnull().any():
            mask = ~y_val.isnull()
            X_val = X_val[mask]
            y_val = y_val[mask]
            logger.warning(f"   ⚠️  Removed {(~mask).sum()} rows with NaN target from val")

        if y_test.isnull().any():
            mask = ~y_test.isnull()
            X_test = X_test[mask]
            y_test = y_test[mask]
            logger.warning(f"   ⚠️  Removed {(~mask).sum()} rows with NaN target from test")

        # Ensure target is integer (0 or 1)
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

        logger.info("   ✅ Target variable cleaned")

        # Handle missing values in features
        X_train, X_val, X_test = self._handle_missing_values(X_train, X_val, X_test)

        # Balance classes if enabled
        if self.data_config.get("balance_classes", False):
            X_train, y_train = self._balance_classes(X_train, y_train)

        # Log class distribution
        self._log_class_distribution(y_train, y_val, y_test)

        logger.info(f"\n✅ Features prepared successfully")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """Identify numeric and categorical features."""
        logger.info(f"\n🔍 Identifying Feature Types:")

        # Get specified features from config
        config_numeric = self.data_config.get("numeric_features", [])
        config_categorical = self.data_config.get("categorical_features", [])

        if config_numeric and config_categorical:
            # Use config-specified types
            self.numeric_features = [f for f in config_numeric if f in self.feature_columns]
            self.categorical_features = [f for f in config_categorical if f in self.feature_columns]
        else:
            # Auto-detect
            for col in self.feature_columns:
                if col not in df.columns:
                    continue

                dtype = df[col].dtype

                if dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Check if it's actually categorical (low cardinality)
                    unique_count = df[col].nunique()
                    if unique_count <= 20 and dtype in ['int64', 'int32']:
                        self.categorical_features.append(col)
                    else:
                        self.numeric_features.append(col)
                else:
                    self.categorical_features.append(col)

        logger.info(f"   Numeric features:     {len(self.numeric_features)}")
        logger.info(f"   Categorical features: {len(self.categorical_features)}")

        # Log examples
        if self.numeric_features:
            logger.info(f"   Numeric examples: {', '.join(self.numeric_features[:5])}")
        if self.categorical_features:
            logger.info(f"   Categorical examples: {', '.join(self.categorical_features[:5])}")

    def _handle_missing_values(
            self,
            X_train: pd.DataFrame,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Handle missing values in features."""
        logger.info(f"\n🔧 Handling Missing Values:")

        missing_train = X_train.isna().sum().sum()
        missing_val = X_val.isna().sum().sum()
        missing_test = X_test.isna().sum().sum()

        logger.info(f"   Train missing: {missing_train:,}")
        logger.info(f"   Val missing:   {missing_val:,}")
        logger.info(f"   Test missing:  {missing_test:,}")

        if missing_train == 0 and missing_val == 0 and missing_test == 0:
            logger.info(f"   ✅ No missing values found")
            return X_train, X_val, X_test

        strategy_config = self.data_config.get("missing_value_strategy", {})
        numeric_strategy = strategy_config.get("numeric", "median")
        categorical_strategy = strategy_config.get("categorical", "mode")

        # 👇 CORREÇÃO: Usar .loc[] e criar cópias explícitas
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        # Handle numeric features
        for col in self.numeric_features:
            if col not in X_train.columns:
                continue

            if X_train[col].isna().any():
                if numeric_strategy == "mean":
                    fill_value = X_train[col].mean()
                elif numeric_strategy == "median":
                    fill_value = X_train[col].median()
                elif numeric_strategy == "zero":
                    fill_value = 0
                else:
                    fill_value = 0

                # Se fill_value for NaN (coluna toda vazia), usar 0
                if pd.isna(fill_value):
                    fill_value = 0

                X_train.loc[:, col] = X_train[col].fillna(fill_value)
                X_val.loc[:, col] = X_val[col].fillna(fill_value)
                X_test.loc[:, col] = X_test[col].fillna(fill_value)

        # Handle categorical features
        for col in self.categorical_features:
            if col not in X_train.columns:
                continue

            if X_train[col].isna().any():
                if categorical_strategy == "mode":
                    mode_values = X_train[col].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else "unknown"
                elif categorical_strategy == "unknown":
                    fill_value = "unknown"
                else:
                    fill_value = "unknown"

                X_train.loc[:, col] = X_train[col].fillna(fill_value)
                X_val.loc[:, col] = X_val[col].fillna(fill_value)
                X_test.loc[:, col] = X_test[col].fillna(fill_value)

        # 👇 NOVO: Preenchimento final forçado para garantir que não há NaN
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

        # Verificação final
        final_train_missing = X_train.isna().sum().sum()
        final_val_missing = X_val.isna().sum().sum()
        final_test_missing = X_test.isna().sum().sum()

        if final_train_missing > 0 or final_val_missing > 0 or final_test_missing > 0:
            logger.warning(f"   ⚠️  Still have NaN after filling: "
                          f"Train={final_train_missing}, Val={final_val_missing}, Test={final_test_missing}")
        else:
            logger.info(f"   ✅ Missing values handled")

        return X_train, X_val, X_test

    def _balance_classes(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes using specified method."""
        logger.info(f"\n⚖️  Balancing Classes:")

        method = self.data_config.get("balance_method", "none")

        if method == "none":
            logger.info(f"   ⚠️  Balancing disabled")
            return X_train, y_train

        original_counts = y_train.value_counts()
        logger.info(f"   Original distribution:")
        logger.info(f"     Class 0: {original_counts.get(0, 0):,}")
        logger.info(f"     Class 1: {original_counts.get(1, 0):,}")

        try:
            if method == "smote":
                # SMOTE oversampling
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"   ✅ Applied SMOTE")

            elif method == "undersample":
                # Random undersampling
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                logger.info(f"   ✅ Applied Random Undersampling")

            elif method == "oversample":
                # Random oversampling
                X_minority = X_train[y_train == 1]
                y_minority = y_train[y_train == 1]
                X_majority = X_train[y_train == 0]
                y_majority = y_train[y_train == 0]

                X_minority_upsampled, y_minority_upsampled = resample(
                    X_minority, y_minority,
                    replace=True,
                    n_samples=len(X_majority),
                    random_state=42
                )

                X_train = pd.concat([X_majority, X_minority_upsampled])
                y_train = pd.concat([y_majority, y_minority_upsampled])
                logger.info(f"   ✅ Applied Random Oversampling")

            new_counts = y_train.value_counts()
            logger.info(f"   New distribution:")
            logger.info(f"     Class 0: {new_counts.get(0, 0):,}")
            logger.info(f"     Class 1: {new_counts.get(1, 0):,}")

        except Exception as e:
            logger.warning(f"   ⚠️  Balancing failed: {e}")
            logger.warning(f"   Continuing with original distribution")

        return X_train, y_train

    def _log_class_distribution(
            self,
            y_train: pd.Series,
            y_val: pd.Series,
            y_test: pd.Series
    ) -> None:
        """Log class distribution for all splits."""
        logger.info(f"\n📊 Class Distribution:")

        for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            counts = y.value_counts()
            total = len(y)

            class_0 = counts.get(0, 0)
            class_1 = counts.get(1, 0)

            logger.info(f"   {name:5s}: Class 0: {class_0:>8,} ({class_0 / total:>6.2%}) | "
                        f"Class 1: {class_1:>8,} ({class_1 / total:>6.2%})")

    def get_feature_info(self) -> Dict[str, List[str]]:
        """
        Get feature information.

        Returns:
            Dictionary with feature lists
        """
        return {
            "all_features": self.feature_columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features
        }