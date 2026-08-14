from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger


logger = get_logger("transaction_features")


def add_transaction_features(events: pd.DataFrame) -> pd.DataFrame:
    

    df = events.copy()

    required = ["event_type", "amount", "currency", "transaction_type", "channel"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column for transaction features: {col}")

    if "transaction_description" not in df.columns:
        df["transaction_description"] = None

    if "beneficiary_bank_code" not in df.columns:
        df["beneficiary_bank_code"] = None

    if "merchant_category_code" not in df.columns:
        df["merchant_category_code"] = None

    if "is_round_amount" not in df.columns:
        df["is_round_amount"] = 0

    if "is_international" not in df.columns:
        df["is_international"] = 0

    if "amount_increase_vs_30d_mean" not in df.columns:
        df["amount_increase_vs_30d_mean"] = np.nan

    tx_mask = df["event_type"] == "transaction"

    df.loc[tx_mask, "is_round_amount"] = (
        df.loc[tx_mask, "amount"].fillna(0.0).apply(lambda x: int(x % 10 == 0 or x % 100 == 0))
    )

    if "registered_country" in df.columns and "recipient_country" in df.columns:
        df.loc[tx_mask, "is_international"] = (
            (df.loc[tx_mask, "registered_country"] != df.loc[tx_mask, "recipient_country"]).astype(int)
        )

    if "amount_mean_last_30d" in df.columns:
        mean_30d = df.loc[tx_mask, "amount_mean_last_30d"].replace(0.0, np.nan)
        current_amount = df.loc[tx_mask, "amount"].fillna(0.0)
        df.loc[tx_mask, "amount_increase_vs_30d_mean"] = (current_amount / mean_30d).fillna(0.0)

    def _generate_description(row):
        if row["event_type"] != "transaction":
            return None
        tx_type = row.get("transaction_type", "unknown")
        return f"{tx_type.upper()} payment"

    df["transaction_description"] = df.apply(_generate_description, axis=1)

    def _generate_bank_code(row):
        if row["event_type"] != "transaction":
            return None
        return f"{np.random.randint(100, 999)}"

    df.loc[tx_mask, "beneficiary_bank_code"] = [
        f"{np.random.randint(100, 999)}" for _ in range(tx_mask.sum())
    ]

    mcc_pool = [5411, 5812, 5999, 6011, 7011, 4121]

    def _assign_mcc(row):
        if row["event_type"] != "transaction":
            return None
        tx_type = row.get("transaction_type", "")
        if "card" in tx_type or "credit" in tx_type or "debit" in tx_type:
            return str(np.random.choice(mcc_pool))
        return None

    df["merchant_category_code"] = df.apply(_assign_mcc, axis=1)

    logger.info("Transaction features (1.4) added")
    return df