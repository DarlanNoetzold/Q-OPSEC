from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from utils import available

def make_models(preprocessor) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    # 1) LogReg
    models["logreg_lbfgs"] = Pipeline([("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced", solver="lbfgs", multi_class="auto"))
    ])

    # 2) Linear SVM (calibrado)
    lin_svc = Pipeline([("pre", preprocessor), ("clf", LinearSVC(class_weight="balanced"))])
    models["linear_svm_calibrated"] = Pipeline([("cal", CalibratedClassifierCV(lin_svc, cv=3))])

    # 3) SVC RBF
    models["svc_rbf"] = Pipeline([("pre", preprocessor),
        ("clf", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced"))
    ])

    # 4) RandomForest
    models["rf"] = Pipeline([("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, class_weight="balanced_subsample", n_jobs=-1))
    ])

    # 5) ExtraTrees
    models["et"] = Pipeline([("pre", preprocessor),
        ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=None, class_weight="balanced", n_jobs=-1))
    ])

    # 6) GradientBoosting
    models["gb"] = Pipeline([("pre", preprocessor),
        ("clf", GradientBoostingClassifier())
    ])

    # 7) HistGradientBoosting
    models["hgb"] = Pipeline([("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(max_depth=None))
    ])

    # 8) MLP
    models["mlp"] = Pipeline([("pre", preprocessor),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,128), activation="relu", max_iter=200, early_stopping=True))
    ])

    # 9) KNN
    models["knn"] = Pipeline([("pre", preprocessor),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))
    ])

    # 10) GaussianNB (precisa densificar)
    # Observação: GaussianNB exige dense; OneHot pode ficar caro. Use com cautela.
    # Aqui simplificamos usando preprocessor que retorna sparse; p/ usar NB, ajuste para densificar.
    # Para manter >=10 configs, deixamos registrado, mas você pode desativar se memória for pouca.
    # models["gnb"] = Pipeline([...])

    # 10) XGBoost (opcional)
    if available("xgboost"):
        from xgboost import XGBClassifier
        models["xgb"] = Pipeline([("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", eval_metric="mlogloss", n_jobs=-1
            ))
        ])

    # 11) LightGBM (opcional)
    if available("lightgbm"):
        import lightgbm as lgb
        models["lgbm"] = Pipeline([("pre", preprocessor),
            ("clf", lgb.LGBMClassifier(
                n_estimators=800, num_leaves=63, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, n_jobs=-1
            ))
        ])

    # 12) CatBoost (opcional, robusto p/ categóricas, mas exige tratamento diferente)
    if available("catboost"):
        from catboost import CatBoostClassifier
        models["cat"] = Pipeline([("pre", preprocessor),
            ("clf", CatBoostClassifier(
                iterations=800, depth=8, learning_rate=0.05, loss_function="MultiClass", verbose=False
            ))
        ])

    return models