"""
Model classes package. Model implementations are imported on demand so loading
one serialized estimator does not require every optional ML framework.
"""

__all__ = [
    "BaseModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "LightGBMModel",
    "XGBoostModel",
]


def __getattr__(name):
    modules = {
        "BaseModel": ".base_model",
        "LogisticRegressionModel": ".logistic_regression_model",
        "RandomForestModel": ".random_forest_model",
        "LightGBMModel": ".lightgbm_model",
        "XGBoostModel": ".xgboost_model",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(module_name, __name__), name)
