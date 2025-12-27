from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger("fraud_history_features")


def add_fraud_history_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.8 Fraud / Abuse / Historical Risk features.

    Fields:
      - previous_fraud_count
      - previous_chargeback_count
      - account_takeover_flag
      - velocity_alert_flag
      - blacklist_hit
      - whitelist_hit
      - money_mule_score
      - device_fingerprint_match_count
      - ip_reputation_score
    """

    df = events.copy()

    # Initialize all fraud history columns with safe defaults
    df["previous_fraud_count"] = 0
    df["previous_chargeback_count"] = 0
    df["account_takeover_flag"] = 0
    df["velocity_alert_flag"] = 0
    df["blacklist_hit"] = 0
    df["whitelist_hit"] = 0
    df["money_mule_score"] = 0.0
    df["device_fingerprint_match_count"] = 0
    df["ip_reputation_score"] = np.random.uniform(0.3, 0.9, size=len(df))

    # Money mule detection - only if we have recipients
    if "recipient_id" in df.columns:
        unique_recipients = df["recipient_id"].dropna().unique()

        if len(unique_recipients) > 0:
            # Sample 1% as potential money mules
            num_mules = max(1, int(len(unique_recipients) * 0.01))
            mule_recipients = np.random.choice(
                unique_recipients,
                size=num_mules,
                replace=False
            )

            # Mark transactions to these recipients
            mask = df["recipient_id"].isin(mule_recipients)
            df.loc[mask, "money_mule_score"] = np.random.uniform(0.6, 0.95, size=mask.sum())
        else:
            logger.warning("No recipients found in dataset, skipping money mule detection")

    # Velocity alerts - based on temporal features if available
    if "transactions_last_1h" in df.columns:
        df.loc[df["transactions_last_1h"] > 10, "velocity_alert_flag"] = 1

    # Blacklist/whitelist - random for now (2% blacklist, 15% whitelist)
    df["blacklist_hit"] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
    df["whitelist_hit"] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])

    # Device fingerprint matches - count how many times each device appears
    if "device_id" in df.columns:
        device_counts = df.groupby("device_id").size()
        df["device_fingerprint_match_count"] = df["device_id"].map(device_counts).fillna(1).astype(int)

    # Previous fraud/chargeback counts - synthetic based on user risk class
    if "user_risk_class" in df.columns:
        high_risk_mask = df["user_risk_class"] == "high"
        df.loc[high_risk_mask, "previous_fraud_count"] = np.random.poisson(2, size=high_risk_mask.sum())
        df.loc[high_risk_mask, "previous_chargeback_count"] = np.random.poisson(1, size=high_risk_mask.sum())

    # Account takeover flag - rare event (0.5%)
    df["account_takeover_flag"] = np.random.choice([0, 1], size=len(df), p=[0.995, 0.005])

    logger.info("Fraud/historical risk features (1.8) added")
    return df