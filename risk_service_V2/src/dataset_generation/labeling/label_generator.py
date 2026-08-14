from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("label_generator")


def apply_labels(events: pd.DataFrame) -> pd.DataFrame:
    

    df = events.copy()

    dataset_cfg = default_config_loader.load("dataset_config.yaml")
    fraud_rate = float(dataset_cfg.get("generation", {}).get("fraud_rate", 0.01))

    df["is_fraud"] = 0
    df["fraud_type"] = None
    df["label_source"] = "normal"
    df["fraud_confidence"] = 0.0

    target_fraud_count = int(len(df) * fraud_rate)
    logger.info(f"Targeting approximately {target_fraud_count} fraudulent events (rate={fraud_rate:.4f})")

    score = np.zeros(len(df))

    if "user_risk_class" in df.columns:
        score += (df["user_risk_class"] == "high").astype(float) * 5.0

    if "velocity_alert_flag" in df.columns:
        score += df["velocity_alert_flag"].fillna(0) * 3.0

    if "blacklist_hit" in df.columns:
        score += df["blacklist_hit"].fillna(0) * 4.0

    if "money_mule_score" in df.columns:
        score += df["money_mule_score"].fillna(0) * 3.0

    tx_mask = df["event_type"] == "transaction"
    if "is_new_recipient" in df.columns:
        score[tx_mask] += df.loc[tx_mask, "is_new_recipient"].fillna(0) * 2.0

    if "amount" in df.columns:
        amounts = df["amount"].fillna(0)
        if amounts.std() > 0:
            amount_zscore = (amounts - amounts.mean()) / amounts.std()
            score += np.clip(amount_zscore, 0, 3)

    if "hour_of_day" in df.columns:
        unusual_hours = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5])
        score += unusual_hours.astype(float) * 1.5

    if "is_vpn" in df.columns:
        score += df["is_vpn"].fillna(0) * 2.0

    if "is_emulator" in df.columns:
        score += df["is_emulator"].fillna(0) * 2.5

    if target_fraud_count > 0:
        fraud_indices = np.argsort(score)[-target_fraud_count:]

        df.loc[fraud_indices, "is_fraud"] = 1
        df.loc[fraud_indices, "label_source"] = "rule_based"
        df.loc[fraud_indices, "fraud_confidence"] = np.random.uniform(0.7, 0.95, size=len(fraud_indices))

        fraud_df_indices = df.index[fraud_indices]

        for idx in fraud_df_indices:
            if "money_mule_score" in df.columns and df.loc[idx, "money_mule_score"] > 0.5:
                df.loc[idx, "fraud_type"] = "money_mule"
            elif "velocity_alert_flag" in df.columns and df.loc[idx, "velocity_alert_flag"] == 1:
                df.loc[idx, "fraud_type"] = "account_takeover"
            elif "blacklist_hit" in df.columns and df.loc[idx, "blacklist_hit"] == 1:
                df.loc[idx, "fraud_type"] = "blacklist"
            else:
                df.loc[idx, "fraud_type"] = np.random.choice([
                    "card_testing",
                    "synthetic_identity",
                    "phishing_induced",
                    "unauthorized_transaction"
                ])

    fraud_count = df["is_fraud"].sum()
    fraud_pct = (fraud_count / len(df)) * 100 if len(df) > 0 else 0

    logger.info(f"Labels applied: {fraud_count} fraudulent events ({fraud_pct:.2f}%)")
    logger.info(f"Fraud types distribution:\n{df[df['is_fraud'] == 1]['fraud_type'].value_counts()}")

    return df