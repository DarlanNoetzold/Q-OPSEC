from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

from utils import available

# Transformer simples para converter matriz esparsa em densa (para GaussianNB)
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def make_models(preprocessor) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    # 1) LogReg (boa baseline e calibrada naturalmente)
    models["logreg_lbfgs"] = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            C=1.5,
        ))
    ])

    # 2) Linear SVM calibrado (probabilidades úteis p/ thresholding)
    lin = Pipeline([("pre", preprocessor), ("clf", LinearSVC(class_weight="balanced"))])
    models["linear_svm_calibrated"] = Pipeline([
        ("cal", CalibratedClassifierCV(lin, cv=3))
    ])

    # 3) SVC RBF (dois sabores)
    models["svc_rbf_std"] = Pipeline([
        ("pre", preprocessor),
        ("clf", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"))
    ])
    models["svc_rbf_tuned"] = Pipeline([
        ("pre", preprocessor),
        ("clf", SVC(C=4.0, kernel="rbf", gamma=0.2, probability=True, class_weight="balanced"))
    ])

    # 4) RandomForest (duas variações)
    models["rf_balanced"] = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            bootstrap=True
        ))
    ])
    models["rf_deeper"] = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=None,
            n_jobs=-1,
            bootstrap=True
        ))
    ])

    # 5) ExtraTrees (duas variações)
    models["et_balanced"] = Pipeline([
        ("pre", preprocessor),
        ("clf", ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
    models["et_fast"] = Pipeline([
        ("pre", preprocessor),
        ("clf", ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    # 6) GradientBoosting (clássico)
    models["gb"] = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8
        ))
    ])

    # 7) HistGradientBoosting (forte em tabular)
    models["hgb_std"] = Pipeline([
        ("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_depth=None,
            l2_regularization=0.0,
            learning_rate=0.06,
            max_bins=255
        ))
    ])
    models["hgb_reg"] = Pipeline([
        ("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_depth=None,
            l2_regularization=0.5,
            learning_rate=0.05,
            max_bins=255
        ))
    ])

    # 8) MLP (duas variações)
    models["mlp_medium"] = Pipeline([
        ("pre", preprocessor),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15
        ))
    ])
    models["mlp_wide"] = Pipeline([
        ("pre", preprocessor),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(512, 256, 64),
            activation="relu",
            alpha=5e-4,
            learning_rate_init=7e-4,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20
        ))
    ])

    # 9) KNN
    models["knn_dist"] = Pipeline([
        ("pre", preprocessor),
        ("clf", KNeighborsClassifier(n_neighbors=17, weights="distance", p=2))
    ])

    # 10) GaussianNB (com densificação)
    models["gnb_dense"] = Pipeline([
        ("pre", preprocessor),
        ("to_dense", DenseTransformer()),
        ("clf", GaussianNB(var_smoothing=1e-9))
    ])

    # 11) XGBoost (opcional)
    if available("xgboost"):
        from xgboost import XGBClassifier
        models["xgb_hist"] = Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=600,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                reg_alpha=0.0,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=-1
            ))
        ])

    # 12) LightGBM (opcional)
    if available("lightgbm"):
        import lightgbm as lgb
        models["lgbm_std"] = Pipeline([
            ("pre", preprocessor),
            ("clf", lgb.LGBMClassifier(
                n_estimators=900,
                num_leaves=63,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=0.0,
                reg_alpha=0.0,
                n_jobs=-1
            ))
        ])

    # 13) CatBoost (opcional)
    if available("catboost"):
        from catboost import CatBoostClassifier
        models["catboost_multi"] = Pipeline([
            ("pre", preprocessor),
            ("clf", CatBoostClassifier(
                iterations=900,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                loss_function="MultiClass",
                verbose=False
            ))
        ])

    return models