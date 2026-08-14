from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from dateutil import tz

from src.common.logger import get_logger


logger = get_logger("event_identification_features")


def add_event_identification_features(events: pd.DataFrame) -> pd.DataFrame:
    """Ensure all 1.1 Event Identification fields exist and are consistent.

    Fields:
      - event_id (kept if exists, otherwise generated)
      - event_type
      - event_source
      - timestamp_utc
      - timestamp_local
      - timezone
    """

    df = events.copy()

    # Ensure event_id
    if "event_id" not in df.columns:
        logger.warning("event_id column missing, generating sequential IDs")
        df["event_id"] = [f"E{i:010d}" for i in range(len(df))]

    # Basic sanity checks
    required_cols = ["event_type", "event_source", "timestamp_utc", "timezone"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for event identification: {col}")

    # Convert timestamp_utc to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Compute timestamp_local using timezone per row
    def _to_local(row) -> Optional[datetime]:
        ts_utc = row["timestamp_utc"]
        tz_name = row["timezone"]
        try:
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.tz_localize("UTC")
            target_tz = tz.gettz(tz_name)
            if target_tz is None:
                return ts_utc
            return ts_utc.astimezone(target_tz)
        except Exception:
            return ts_utc

    df["timestamp_local"] = df.apply(_to_local, axis=1)

    logger.info("Event identification features added: timestamp_local computed")
    return df