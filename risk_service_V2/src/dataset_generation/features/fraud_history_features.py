from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger("fraud_history_features")


def add_fraud_history_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.8 Fraud / Abuse / Historical Risk features.

    Fields:
      - user_fraud_history_count (total frauds for this user in the past)
      - ip_fraud_history_count (total frauds for this IP in the past)
      - device_fraud_history_count (total frauds for this device in the past)
      - recipient_fraud_history_count (total frauds for this recipient in the past)
      - is_blacklisted_user (placeholder)
      - is_blacklisted_recipient (placeholder)
      - risk_score_historical (0-1, synthetic based on counts)

    Note: This module assumes 'is_fraud' label might be assigned later or
    simulated based on scenarios. For the generation phase, we simulate
    historical counts based on the user profile and scenario.
    """

    df = events.copy()

    # Ensure required columns for grouping
    for col in ["user_id", "ip_address", "device_id", "recipient_id"]:
        if col not in df.columns:
            df[col] = None

    # Initialize columns
    df["user_fraud_history_count"] = 0
    df["ip_fraud_history_count"] = 0
    df["device_fraud_history_count"] = 0
    df["recipient_fraud_history_count"] = 0
    df["is_blacklisted_user"] = 0
    df["is_blacklisted_recipient"] = 0
    df["risk_score_historical"] = 0.0

    # In a real dataset generation, we would look at 'is_fraud' labels of PREVIOUS events.
    # Since we are generating the dataset now, we simulate these counts based on
    # the user's risk class or profile.

    # Heuristic simulation:
    # High risk users/profiles get higher historical counts
    if "user_risk_class" in df.columns:
        high_risk_mask = df["user_risk_class"] == "high"
        # Randomly assign 1-3 historical frauds to some high risk users
        df.loc[high_risk_mask, "user_fraud_history_count"] = np.random.choice([0, 1, 2, 3], size=high_risk_mask.sum(),
                                                                              p=[0.7, 0.2, 0.07, 0.03])

    # IP/Device/Recipient counts:
    # If IP is blacklisted (from geo_utils/network_catalog), set a high count
    if "is_blacklisted_ip" in df.columns:
        df.loc[df["is_blacklisted_ip"] == 1, "ip_fraud_history_count"] = np.random.randint(5, 50, size=(
                    df["is_blacklisted_ip"] == 1).sum())

    # Recipient history:
    # Some recipients are "mules" or known fraudsters
    if "recipient_id" in df.columns:
        # Simulate: 1% of recipients have a history
        unique_recipients = df["recipient_id"].dropna().unique()
        mule_recipients = np.random.choice(unique_recipients, size=max(1, int(len(unique_recipients) * 0.01)),
                                           replace=False)
        df.loc[df["recipient_id"].isin(mule_recipients), "recipient_fraud_history_count"] = np.random.randint(1, 10,
                                                                                                              size=df[
                                                                                                                  "recipient_id"].isin(
                                                                                                                  mule_recipients).sum())
        df.loc[df["recipient_id"].isin(mule_recipients), "is_blacklisted_recipient"] = 1

    # Compute risk_score_historical (0-1)
    # Simple weighted sum normalized
    total_counts = (
            df["user_fraud_history_count"] * 2.0 +
            df["ip_fraud_history_count"] * 1.0 +
            df["device_fraud_history_count"] * 1.5 +
            df["recipient_fraud_history_count"] * 2.0
    )
    # Normalize to 0-1 range (clipping at a reasonable max like 20)
    df["risk_score_historical"] = (total_counts / 20.0).clip(0, 1)

    logger.info("Fraud history and historical risk features (1.8) added")
    return df