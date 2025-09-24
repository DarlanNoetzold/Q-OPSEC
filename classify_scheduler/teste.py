import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

N = 1000
labels = ["Very Low", "Low", "Medium", "High", "Very High", "Critical"]

records = []
for i in range(N):
    label = random.choice(labels)

    # score coerente com o rótulo
    if label == "Very Low": score = np.random.uniform(0, 0.1)
    elif label == "Low": score = np.random.uniform(0.1, 0.3)
    elif label == "Medium": score = np.random.uniform(0.3, 0.5)
    elif label == "High": score = np.random.uniform(0.5, 0.7)
    elif label == "Very High": score = np.random.uniform(0.7, 0.9)
    else: score = np.random.uniform(0.9, 1.0)

    created_at = datetime.now() - timedelta(minutes=i)
    hour = created_at.hour
    day = created_at.weekday()
    month = created_at.month
    year = created_at.year

    record = {
        "id": i,
        "request_id_resolved": f"req_{i}",
        "created_at": created_at.isoformat(),
        "risk_score": score,
        "conf_score": score * np.random.uniform(0.7, 1.2),
        "combined_score": score * np.random.uniform(0.8, 1.1),
        "risk_level": label if label in ["Low", "Medium", "High", "Critical"] else "medium",
        "conf_classification": random.choice(["public", "internal", "confidential", "restricted"]),
        "src_geo": random.choice(["US", "EU", "BR", "JP", "RU"]),
        "src_device_type": random.choice(["mobile", "desktop", "server", "iot"]),
        "dst_service_type": random.choice(["database", "api", "web", "queue"]),
        "dst_security_policy": random.choice(["low", "medium", "high"]),
        "src_mfa_status_norm": random.choice(["enabled","disabled","unknown"]),
        "hour_of_day": hour,
        "day_of_week": day,
        "month": month,
        "year": year,
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "day_sin": np.sin(2*np.pi*day/7),
        "day_cos": np.cos(2*np.pi*day/7),
        "security_level_label": label
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("synthetic_context_dataset.csv", index=False)