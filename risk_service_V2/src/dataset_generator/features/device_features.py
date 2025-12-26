from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger


logger = get_logger("device_environment_features")


def add_device_environment_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.6 Device / Environment features.

    Fields (mostly synthetic/derived at this stage):
      - device_id
      - device_type (mobile, desktop, tablet, etc.)
      - os_family (Android, iOS, Windows, macOS, Linux, etc.)
      - os_version
      - browser_family
      - browser_version
      - app_version
      - is_emulator
      - is_rooted_or_jailbroken
      - is_device_compromised (placeholder heuristic)
      - is_new_device_for_user
      - devices_last_30d
    """

    df = events.copy()

    # Garantir colunas básicas (podem ter vindo do gerador de eventos/perfis)
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

    # Ordenar por usuário e tempo para lógicas de histórico
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    df = df.sort_values(["user_id", "timestamp_utc"]).reset_index(drop=True)

    # Novos campos
    df["is_new_device_for_user"] = 0
    df["devices_last_30d"] = 0
    df["is_device_compromised"] = 0  # placeholder, pode usar fraude/cenário depois

    # Cálculo por usuário
    def _per_user(user_df: pd.DataFrame) -> pd.DataFrame:
        user_df = user_df.copy()

        # is_new_device_for_user via duplicated
        user_df["is_new_device_for_user"] = (
            ~user_df["device_id"].astype(str).duplicated()
        ).astype(int)

        # devices_last_30d: número de devices distintos nos últimos 30 dias
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

        # Heurística simples para is_device_compromised:
        # marcar 1 se is_emulator ou is_rooted_or_jailbroken, ou se device_type == "unknown" e evento suspeito
        compromised = (
            (user_df["is_emulator"].astype(int) == 1)
            | (user_df["is_rooted_or_jailbroken"].astype(int) == 1)
        )
        user_df.loc[compromised, "is_device_compromised"] = 1

        return user_df

    df = df.groupby("user_id", group_keys=False).apply(_per_user)

    logger.info("Device/Environment features (1.6) added")
    return df