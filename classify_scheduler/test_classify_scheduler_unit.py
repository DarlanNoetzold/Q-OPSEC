import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from configs import DefaultConfig, SECURITY_ORDER, CANONICAL, ALLOWED_CLASSES
from utils import (
    to_canonical,
    seed_everything,
    normalize_labels,
    ensure_classes,
    validate_required_columns,
    create_feature_mapping,
    get_feature_columns_from_df,
    format_class_mapping,
    validate_security_levels,
    available,
)
from preprocessing import build_preprocessor, DROP_COLS
from models_zoo import DenseTransformer, make_models
from data import load_dataset


class ClassifySchedulerUnitTests(unittest.TestCase):
    def test_to_canonical_standard_and_variations(self):
        self.assertEqual("Very Low", to_canonical("very low"))
        self.assertEqual("Very Low", to_canonical("verylow"))
        self.assertEqual("Very Low", to_canonical("very_low"))
        self.assertEqual("Very Low", to_canonical(" VERY LOW "))
        self.assertEqual("Low", to_canonical("low"))
        self.assertEqual("Medium", to_canonical("medium"))
        self.assertEqual("High", to_canonical("high"))
        self.assertEqual("Very High", to_canonical("very high"))
        self.assertEqual("Very High", to_canonical("veryhigh"))
        self.assertEqual("Critical", to_canonical("critical"))
        self.assertEqual("", to_canonical("unknown_label"))
        self.assertEqual("", to_canonical(None))
        self.assertEqual("", to_canonical(123))

    def test_seed_everything_ensures_deterministic_random_choice(self):
        seed_everything(1234)
        val_a = np.random.rand()
        seed_everything(1234)
        val_b = np.random.rand()
        self.assertEqual(val_a, val_b)

    def test_normalize_labels_drops_invalid_and_maps_valid(self):
        df = pd.DataFrame({
            "security_level_label": ["critical", "low", "invalid_label", "  medium  ", ""],
            "feature": [1, 2, 3, 4, 5]
        })
        cleaned_df, dropped_count = normalize_labels(df, "security_level_label")
        self.assertEqual(2, dropped_count)
        self.assertEqual(3, len(cleaned_df))
        self.assertListEqual(
            ["Critical", "Low", "Medium"],
            cleaned_df["security_level_label"].tolist()
        )

    def test_ensure_classes_accepts_multiple_classes(self):
        df = pd.DataFrame({"target": ["Critical", "Low", "Medium"]})
        # Should not raise
        ensure_classes(df, "target", ALLOWED_CLASSES)

    def test_ensure_classes_rejects_single_class(self):
        df = pd.DataFrame({"target": ["Critical", "Critical"]})
        with self.assertRaises(ValueError) as ctx:
            ensure_classes(df, "target", ALLOWED_CLASSES)
        self.assertIn("Need at least 2 classes", str(ctx.exception))

    def test_validate_required_columns(self):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        missing = validate_required_columns(df, ["col_a", "col_b", "col_c"])
        self.assertEqual(["col_c"], missing)

        no_missing = validate_required_columns(df, ["col_a", "col_b"])
        self.assertEqual([], no_missing)

    def test_create_feature_mapping(self):
        mapping = create_feature_mapping()
        self.assertIn("temporal_features", mapping)
        self.assertIn("risk_features", mapping)
        self.assertIn("categorical_features", mapping)
        self.assertIn("identifier_features", mapping)

    def test_get_feature_columns_from_df(self):
        df = pd.DataFrame({
            "id": [1],
            "request_id_resolved": ["req_1"],
            "risk_score": [0.5],
            "src_geo": ["US"],
            "security_level_label": ["Critical"]
        })
        cols = get_feature_columns_from_df(df, "security_level_label", exclude_identifiers=True)
        self.assertIn("risk_score", cols)
        self.assertIn("src_geo", cols)
        self.assertNotIn("id", cols)
        self.assertNotIn("request_id_resolved", cols)
        self.assertNotIn("security_level_label", cols)

        cols_with_ids = get_feature_columns_from_df(df, "security_level_label", exclude_identifiers=False)
        self.assertIn("id", cols_with_ids)
        self.assertIn("request_id_resolved", cols_with_ids)

    def test_format_class_mapping(self):
        mapping = format_class_mapping(["Low", "Medium", "High"])
        self.assertEqual({0: "Low", 1: "Medium", 2: "High"}, mapping)

    def test_validate_security_levels(self):
        self.assertTrue(validate_security_levels(["Critical", "Very High", "High"]))
        self.assertFalse(validate_security_levels(["Low", "Critical"]))

    def test_available_helper(self):
        self.assertTrue(available("sys"))
        self.assertFalse(available("non_existent_library_xyz_123"))

    def test_dense_transformer(self):
        transformer = DenseTransformer()
        transformer.fit(None)
        dense_arr = np.array([[1, 2], [3, 4]])
        result = transformer.transform(dense_arr)
        np.testing.assert_array_equal(dense_arr, result)

    def test_build_preprocessor_identifies_numeric_and_categorical(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "request_id_resolved": ["r1", "r2", "r3"],
            "created_at": ["t1", "t2", "t3"],
            "encryption_script_label": ["s1", "s2", "s3"],
            "processing_priority_label": ["p1", "p2", "p3"],
            "risk_score": [0.1, 0.5, 0.9],
            "is_flagged": [True, False, True],
            "src_geo": ["US", "BR", "JP"],
            "target": ["Low", "Medium", "High"]
        })
        pre, feat_cols = build_preprocessor(df, target_col="target")
        self.assertIn("risk_score", feat_cols)
        self.assertIn("is_flagged", feat_cols)
        self.assertIn("src_geo", feat_cols)
        for drop_col in DROP_COLS:
            self.assertNotIn(drop_col, feat_cols)

    def test_models_zoo_pipeline_construction(self):
        df = pd.DataFrame({
            "risk_score": [0.1, 0.5, 0.9],
            "src_geo": ["US", "BR", "JP"],
        })
        pre, _ = build_preprocessor(df, target_col=None)
        models = make_models(pre)
        self.assertIn("logreg_lbfgs", models)
        self.assertIn("rf_balanced", models)
        self.assertIn("hgb_std", models)
        self.assertIn("knn_dist", models)
        self.assertIn("gnb_dense", models)

    def test_default_config_instantiation(self):
        cfg = DefaultConfig(
            data_path=Path("data.csv"),
            registry_dir=Path("registry"),
            seed=42,
            test_size=0.2,
            cv_splits=5,
            max_models=10,
            target_col="security_level_label",
        )
        self.assertEqual(42, cfg.seed)
        self.assertEqual(5, cfg.cv_splits)
        self.assertEqual(SECURITY_ORDER, cfg.allowed_classes)


if __name__ == "__main__":
    unittest.main()
