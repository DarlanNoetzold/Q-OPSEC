from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger


logger = get_logger("device_environment_features")


def add_device_environment_features(events: pd.DataFrame) -> pd.DataFrame:
    

    df = events.copy()

    base_cols_defaults = {
        "device_id": None,
        "device_type": "unknown",
        "os_family": "unknown",
        "os_version": None,
        "browser_family": None,
        "browser_version": None,
        "app_version": None,
        "is_emulator": 0,
        "is_rooted_or_jailbroken": 0,
    }

    for col, default in base_cols_defaults.items():
        if col not in df.columns:
            df[col] = default

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    df = df.sort_values(["user_id", "timestamp_utc"]).reset_index(drop=True)

    df["is_new_device_for_user"] = 0
    df["devices_last_30d"] = 0
    df["is_device_compromised"] = 0

    def _per_user(user_df: pd.DataFrame) -> pd.DataFrame:
        user_df = user_df.copy()

        if user_df.empty:
            return user_df

        if "user_id" not in user_df.columns:
            if user_df.index.name == "user_id":
                user_df = user_df.reset_index()
            elif "user_id" in user_df.index.names:
                user_df = user_df.reset_index()

        user_df["is_new_device_for_user"] = (
            ~user_df["device_id"].astype(str).duplicated()
        ).astype(int)

        times = user_df["timestamp_utc"].values.astype("datetime64[s]").astype("int64")
        device_ids = user_df["device_id"].astype(str).values

        devices_last_30d = []
        for i in range(len(user_df)):
            t = times[i]
            win_30d = t - 30 * 86400

            prev_idx = slice(0, i)
            t_prev = times[prev_idx]
            d_prev = device_ids[prev_idx]

            mask_30d = (t_prev >= win_30d) & (t_prev < t)
            devices_last_30d.append(len(set(d_prev[mask_30d])))

        user_df["devices_last_30d"] = devices_last_30d

        compromised = (
            (user_df["is_emulator"].astype(int) == 1)
            | (user_df["is_rooted_or_jailbroken"].astype(int) == 1)
        )
        user_df.loc[compromised, "is_device_compromised"] = 1

        return user_df

    df = df.groupby("user_id", group_keys=False).apply(_per_user)

    logger.info("Device/Environment features (1.6) added")
    return df