from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.common.logger import get_logger


logger = get_logger("temporal_features")


def add_temporal_behavioral_features(events: pd.DataFrame) -> pd.DataFrame:
    

    df = events.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    df = df.sort_values(["user_id", "timestamp_utc"]).reset_index(drop=True)

    df["hour_of_day"] = df["timestamp_utc"].dt.hour
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["date_local"] = df["timestamp_utc"].dt.date
    df["is_local_holiday"] = df["date_local"].isin(
        [
            datetime(2024, 1, 1).date(),
            datetime(2024, 12, 25).date(),
        ]
    ).astype(int)

    df["seconds_since_last_login"] = np.nan
    df["seconds_since_last_transaction"] = np.nan

    df["transactions_last_1h"] = 0
    df["transactions_last_24h"] = 0
    df["transactions_last_7d"] = 0
    df["transactions_last_30d"] = 0

    df["amount_sum_last_24h"] = 0.0
    df["amount_sum_last_7d"] = 0.0
    df["amount_sum_last_30d"] = 0.0
    df["amount_mean_last_30d"] = 0.0
    df["amount_std_last_30d"] = 0.0

    df["logins_last_24h"] = 0
    df["login_failures_last_24h"] = 0
    df["password_resets_last_30d"] = 0


    df = df.reset_index(drop=True)

    user_ids = df["user_id"].unique()
    processed_chunks = []

    for uid in user_ids:
        user_chunk = df[df["user_id"] == uid].copy()
        if not user_chunk.empty:
            processed_chunk = _compute_user_temporal_features(user_chunk)
            processed_chunks.append(processed_chunk)

    if processed_chunks:
        df = pd.concat(processed_chunks, ignore_index=True)

    potential_garbage = ["index", "level_0", "level_1"]
    for col in potential_garbage:
        if col in df.columns and col != "user_id":
             df = df.drop(columns=[col])

    df = df.drop(columns=["date_local"], errors="ignore")

    logger.info("Temporal/behavioral features (1.3) added")
    return df


def _compute_user_temporal_features(user_df: pd.DataFrame) -> pd.DataFrame:
    

    user_df = user_df.sort_values("timestamp_utc").reset_index(drop=True)

    timestamps = user_df["timestamp_utc"].values.astype("datetime64[s]")
    timestamps_sec = timestamps.astype("int64")

    last_login_time: float | None = None
    last_tx_time: float | None = None

    seconds_since_last_login = []
    seconds_since_last_tx = []

    tx_last_1h = []
    tx_last_24h = []
    tx_last_7d = []
    tx_last_30d = []

    amt_sum_24h = []
    amt_sum_7d = []
    amt_sum_30d = []
    amt_mean_30d = []
    amt_std_30d = []

    logins_24h = []
    login_fail_24h = []
    pwd_reset_30d = []

    amounts = user_df.get("amount", pd.Series([np.nan] * len(user_df))).fillna(0.0).values
    event_types = user_df["event_type"].values

    for i in range(len(user_df)):
        t = timestamps_sec[i]
        etype = event_types[i]
        amt = float(amounts[i]) if not np.isnan(amounts[i]) else 0.0

        if last_login_time is None:
            seconds_since_last_login.append(np.nan)
        else:
            seconds_since_last_login.append(float(t - last_login_time))

        if last_tx_time is None:
            seconds_since_last_tx.append(np.nan)
        else:
            seconds_since_last_tx.append(float(t - last_tx_time))

        win_1h = t - 3600
        win_24h = t - 86400
        win_7d = t - 7 * 86400
        win_30d = t - 30 * 86400

        idx_prev = slice(0, i)
        t_prev = timestamps_sec[idx_prev]
        et_prev = event_types[idx_prev]
        amt_prev = amounts[idx_prev]

        def _window_mask(start_sec: float):
            return (t_prev >= start_sec) & (t_prev < t)

        is_tx_prev = et_prev == "transaction"

        mask_1h = _window_mask(win_1h) & is_tx_prev
        mask_24h = _window_mask(win_24h) & is_tx_prev
        mask_7d = _window_mask(win_7d) & is_tx_prev
        mask_30d = _window_mask(win_30d) & is_tx_prev

        tx_last_1h.append(int(mask_1h.sum()))
        tx_last_24h.append(int(mask_24h.sum()))
        tx_last_7d.append(int(mask_7d.sum()))
        tx_last_30d.append(int(mask_30d.sum()))

        amt_sum_24h.append(float(amt_prev[mask_24h].sum()) if mask_24h.any() else 0.0)
        amt_sum_7d.append(float(amt_prev[mask_7d].sum()) if mask_7d.any() else 0.0)
        amt_sum_30d.append(float(amt_prev[mask_30d].sum()) if mask_30d.any() else 0.0)

        if mask_30d.any():
            vals_30d = amt_prev[mask_30d]
            amt_mean_30d.append(float(vals_30d.mean()))
            amt_std_30d.append(float(vals_30d.std(ddof=0)))
        else:
            amt_mean_30d.append(0.0)
            amt_std_30d.append(0.0)

        is_login_prev = et_prev == "login"
        mask_login_24h = _window_mask(win_24h) & is_login_prev
        mask_pwd_30d = (t_prev >= win_30d) & (t_prev < t) & (et_prev == "password_change")

        logins_24h.append(int(mask_login_24h.sum()))

        login_fail_24h.append(0)
        pwd_reset_30d.append(int(mask_pwd_30d.sum()))

        if etype == "login":
            last_login_time = t
        if etype == "transaction":
            last_tx_time = t

    user_df["seconds_since_last_login"] = seconds_since_last_login
    user_df["seconds_since_last_transaction"] = seconds_since_last_tx

    user_df["transactions_last_1h"] = tx_last_1h
    user_df["transactions_last_24h"] = tx_last_24h
    user_df["transactions_last_7d"] = tx_last_7d
    user_df["transactions_last_30d"] = tx_last_30d

    user_df["amount_sum_last_24h"] = amt_sum_24h
    user_df["amount_sum_last_7d"] = amt_sum_7d
    user_df["amount_sum_last_30d"] = amt_sum_30d
    user_df["amount_mean_last_30d"] = amt_mean_30d
    user_df["amount_std_last_30d"] = amt_std_30d

    user_df["logins_last_24h"] = logins_24h
    user_df["login_failures_last_24h"] = login_fail_24h
    user_df["password_resets_last_30d"] = pwd_reset_30d

    return user_df