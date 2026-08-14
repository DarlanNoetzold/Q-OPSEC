"""
Feature Engineering - Transform and prepare features for modeling
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pickle
from pathlib import Path

from src.common.logger import logger


class FeatureEngineer:
    """Transform and engineer features for model training."""

    def __init__(self, config: dict, feature_info: Optional[Dict] = None):
        """
        Initialize FeatureEngineer.

        Args:
            config: Training configuration dictionary
            feature_info: Dictionary with feature lists (numeric, categorical)
        """
        self.config = config
        self.fe_config = config.get("feature_engineering", {})

        # Handle feature_info
        if feature_info is None:
            feature_info = {}
        self.feature_info = feature_info

        self.numeric_features = feature_info.get("numeric_features", [])
        self.categorical_features = feature_info.get("categorical_features", [])

        # Transformers
        self.scaler = None
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None

        logger.info("FeatureEngineer initialized")

    def fit_transform(
            self,
            X_train: pd.DataFrame,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit transformers on training data and transform all splits.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training target (optional, needed for feature selection)

        Returns:
            Tuple of (X_train_transformed, X_val_transformed, X_test_transformed)
        """
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 80)

        # Auto-detect feature types if not provided
        if not self.numeric_features and not self.categorical_features:
            self._identify_feature_types(X_train)

        # 1. Encode categorical features
        X_train, X_val, X_test = self._encode_categorical(X_train, X_val, X_test)

        # 2. Scale numeric features
        X_train, X_val, X_test = self._scale_numeric(X_train, X_val, X_test)

        # 3. Feature selection (optional)
        if self.fe_config.get("feature_selection", {}).get("enabled", False):
            if y_train is not None:
                X_train, X_val, X_test = self._select_features(X_train, y_train, X_val, X_test)
            else:
                logger.warning("   ⚠️  Feature selection enabled but y_train not provided, skipping")

        logger.info(f"\n✅ Feature engineering completed")
        logger.info(f"   Final feature count: {X_train.shape[1]}")

        return X_train, X_val, X_test

    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Identify numeric and categorical features."""
        logger.info(f"\n🔍 Auto-detecting feature types:")

        self.numeric_features = []
        self.categorical_features = []

        for col in X.columns:
            dtype = X[col].dtype

            if dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if it's actually categorical (low cardinality)
                unique_count = X[col].nunique()
                if unique_count <= 20 and dtype in ['int64', 'int32']:
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)

        logger.info(f"   Numeric features:     {len(self.numeric_features)}")
        logger.info(f"   Categorical features: {len(self.categorical_features)}")

    def _encode_categorical(
            self,
            X_train: pd.DataFrame,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Encode categorical features."""
        if not self.categorical_features:
            logger.info(f"\n📝 Categorical Encoding: No categorical features to encode")
            return X_train, X_val, X_test

        logger.info(f"\n📝 Categorical Encoding:")

        encoding_method = self.fe_config.get("categorical_encoding", "label")
        logger.info(f"   Method: {encoding_method}")
        logger.info(f"   Features to encode: {len(self.categorical_features)}")

        # Filter to existing columns
        cat_features = [f for f in self.categorical_features if f in X_train.columns]

        # Create copies to avoid SettingWithCopyWarning
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        if encoding_method == "label":
            # Label encoding
            for col in cat_features:
                le = LabelEncoder()

                # Fit on train
                X_train.loc[:, col] = X_train[col].astype(str)
                le.fit(X_train[col])

                # Transform all splits
                X_train.loc[:, col] = le.transform(X_train[col])

                # Handle unseen categories in val/test
                X_val.loc[:, col] = X_val[col].astype(str)
                X_val.loc[:, col] = X_val[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

                X_test.loc[:, col] = X_test[col].astype(str)
                X_test.loc[:, col] = X_test[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

                self.label_encoders[col] = le

            logger.info(f"   ✅ Label encoding applied to {len(cat_features)} features")

        elif encoding_method == "onehot":
            # One-hot encoding
            max_categories = self.fe_config.get("max_categories_onehot", 10)

            cols_to_encode = []
            for col in cat_features:
                if X_train[col].nunique() <= max_categories:
                    cols_to_encode.append(col)

            if cols_to_encode:
                # Get dummies
                X_train = pd.get_dummies(X_train, columns=cols_to_encode, drop_first=True)
                X_val = pd.get_dummies(X_val, columns=cols_to_encode, drop_first=True)
                X_test = pd.get_dummies(X_test, columns=cols_to_encode, drop_first=True)

                # Align columns
                X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
                X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

                logger.info(f"   ✅ One-hot encoding applied to {len(cols_to_encode)} features")
                logger.info(f"   New feature count: {X_train.shape[1]}")

            # Label encode remaining high-cardinality features
            remaining_cat = [f for f in cat_features if f not in cols_to_encode and f in X_train.columns]
            if remaining_cat:
                for col in remaining_cat:
                    le = LabelEncoder()
                    X_train.loc[:, col] = X_train[col].astype(str)
                    le.fit(X_train[col])
                    X_train.loc[:, col] = le.transform(X_train[col])

                    X_val.loc[:, col] = X_val[col].astype(str)
                    X_val.loc[:, col] = X_val[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

                    X_test.loc[:, col] = X_test[col].astype(str)
                    X_test.loc[:, col] = X_test[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

                    self.label_encoders[col] = le

                logger.info(f"   ✅ Label encoding applied to {len(remaining_cat)} high-cardinality features")

        elif encoding_method == "target":
            # Target encoding (mean encoding)
            logger.warning(f"   ⚠️  Target encoding not fully implemented, using label encoding")

            for col in cat_features:
                le = LabelEncoder()
                X_train.loc[:, col] = X_train[col].astype(str)
                le.fit(X_train[col])
                X_train.loc[:, col] = le.transform(X_train[col])

                X_val.loc[:, col] = X_val[col].astype(str)
                X_val.loc[:, col] = X_val[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

                X_test.loc[:, col] = X_test[col].astype(str)
                X_test.loc[:, col] = X_test[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

                self.label_encoders[col] = le

        return X_train, X_val, X_test

    def _scale_numeric(
            self,
            X_train: pd.DataFrame,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale numeric features."""
        if not self.numeric_features:
            logger.info(f"\n📏 Numeric Scaling: No numeric features to scale")
            return X_train, X_val, X_test

        logger.info(f"\n📏 Numeric Scaling:")

        scaling_method = self.fe_config.get("numeric_scaling", "standard")
        logger.info(f"   Method: {scaling_method}")

        if scaling_method == "none":
            logger.info(f"   ⚠️  Scaling disabled")
            return X_train, X_val, X_test

        # Filter to existing columns
        num_features = [f for f in self.numeric_features if f in X_train.columns]
        logger.info(f"   Features to scale: {len(num_features)}")

        if not num_features:
            return X_train, X_val, X_test

        # Create copies
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        # Select scaler
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            logger.warning(f"   ⚠️  Unknown scaling method: {scaling_method}, using standard")
            self.scaler = StandardScaler()

        # Fit and transform
        X_train.loc[:, num_features] = self.scaler.fit_transform(X_train[num_features])
        X_val.loc[:, num_features] = self.scaler.transform(X_val[num_features])
        X_test.loc[:, num_features] = self.scaler.transform(X_test[num_features])

        logger.info(f"   ✅ Scaling applied to {len(num_features)} features")

        return X_train, X_val, X_test

    def _select_features(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Select top features based on importance or correlation."""
        logger.info(f"\n🎯 Feature Selection:")

        fs_config = self.fe_config.get("feature_selection", {})
        method = fs_config.get("method", "mutual_info")
        top_k = fs_config.get("top_k", 50)

        logger.info(f"   Method: {method}")
        logger.info(f"   Top K: {top_k}")
        logger.info(f"   Current features: {X_train.shape[1]}")

        if X_train.shape[1] <= top_k:
            logger.info(f"   ⚠️  Feature count already <= {top_k}, skipping selection")
            return X_train, X_val, X_test

        if method == "mutual_info":
            # Mutual information
            selector = SelectKBest(mutual_info_classif, k=min(top_k, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train, y_train)

            # Get selected feature names
            selected_mask = selector.get_support()
            self.selected_features = X_train.columns[selected_mask].tolist()

            X_train = pd.DataFrame(X_train_selected, columns=self.selected_features, index=X_train.index)
            X_val = X_val[self.selected_features]
            X_test = X_test[self.selected_features]

            logger.info(f"   ✅ Selected {len(self.selected_features)} features using mutual information")

        elif method == "correlation":
            # Remove highly correlated features
            corr_threshold = fs_config.get("correlation_threshold", 0.95)

            corr_matrix = X_train.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_threshold)]

            X_train = X_train.drop(columns=to_drop)
            X_val = X_val.drop(columns=to_drop)
            X_test = X_test.drop(columns=to_drop)

            self.selected_features = X_train.columns.tolist()

            logger.info(f"   ✅ Removed {len(to_drop)} highly correlated features (threshold: {corr_threshold})")
            logger.info(f"   Remaining features: {len(self.selected_features)}")

        else:
            logger.warning(f"   ⚠️  Unknown feature selection method: {method}")

        return X_train, X_val, X_test

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders and scalers.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        X = X.copy()

        # Encode categorical
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X.loc[:, col] = X[col].astype(str)
                X.loc[:, col] = X[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Scale numeric
        if self.scaler is not None:
            num_features = [f for f in self.numeric_features if f in X.columns]
            if num_features:
                X.loc[:, num_features] = self.scaler.transform(X[num_features])

        # Select features
        if self.selected_features is not None:
            X = X[self.selected_features]

        return X

    def save_preprocessor(self, output_dir: Path, version: str) -> None:
        """Save fitted preprocessors."""
        preprocessor_dir = output_dir / version
        preprocessor_dir.mkdir(parents=True, exist_ok=True)

        # Save scaler
        if self.scaler is not None:
            scaler_path = preprocessor_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"   💾 Scaler saved to {scaler_path}")

        # Save label encoders
        if self.label_encoders:
            encoders_path = preprocessor_dir / "label_encoders.pkl"
            with open(encoders_path, "wb") as f:
                pickle.dump(self.label_encoders, f)
            logger.info(f"   💾 Label encoders saved to {encoders_path}")

        # Save selected features
        if self.selected_features:
            features_path = preprocessor_dir / "selected_features.pkl"
            with open(features_path, "wb") as f:
                pickle.dump(self.selected_features, f)
            logger.info(f"   💾 Selected features saved to {features_path}")

    def load_preprocessor(self, preprocessor_dir: Path) -> None:
        """Load fitted preprocessors."""
        # Load scaler
        scaler_path = preprocessor_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"   📂 Scaler loaded from {scaler_path}")

        # Load label encoders
        encoders_path = preprocessor_dir / "label_encoders.pkl"
        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"   📂 Label encoders loaded from {encoders_path}")

        # Load selected features
        features_path = preprocessor_dir / "selected_features.pkl"
        if features_path.exists():
            with open(features_path, "rb") as f:
                self.selected_features = pickle.load(f)
            logger.info(f"   📂 Selected features loaded from {features_path}")