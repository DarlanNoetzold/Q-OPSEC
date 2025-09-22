from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_holdout(pipeline, X_test, y_test, allowed_classes: List[str]) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("[HOLDOUT] Classification report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "confusion_matrix": cm.tolist()
    }