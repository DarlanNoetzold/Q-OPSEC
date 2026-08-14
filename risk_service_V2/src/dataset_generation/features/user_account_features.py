from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.common.logger import get_logger


logger = get_logger("user_account_features")


def add_user_account_features(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    

    df = events.copy()

    users_df = users.copy()
    if not pd.api.types.is_datetime64_any_dtype(users_df["account_creation_date"]):
        users_df["account_creation_date"] = pd.to_datetime(users_df["account_creation_date"])

    join_cols = [
      "user_type",
      "user_segment",
      "user_risk_class",
      "registered_country",
      "registered_region",
      "account_creation_date",
    ]
    users_meta = users_df[["user_id"] + join_cols]

    df = df.merge(users_meta, on="user_id", how="left", validate="m:1")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    df["account_age_days"] = (
        (df["timestamp_utc"].dt.normalize() - df["account_creation_date"].dt.normalize())
        .dt.days
        .clip(lower=0)
    )

    if "sensitive_data_change_last_7d" not in df.columns:
        df["sensitive_data_change_last_7d"] = 0

    if "registered_devices_count" not in df.columns:
        df["registered_devices_count"] = 0

    if "active_devices_last_30d" not in df.columns:
        df["active_devices_last_30d"] = 0

    logger.info("User/account features added to events (account_age_days, metadata)")
    return df