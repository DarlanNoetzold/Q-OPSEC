from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("text_features")


def add_text_message_features(df: pd.DataFrame) -> pd.DataFrame:
    
    fraud_config = default_config_loader.load("fraud_scenarios.yaml")
    templates = fraud_config.get("message_templates", {})

    df["message_text"] = None
    df["message_length"] = 0
    df["message_language"] = "en"
    df["contains_url"] = False
    df["contains_phone"] = False
    df["num_special_chars"] = 0

    text_event_types = ["transaction", "message", "login"]
    text_mask = df["event_type"].isin(text_event_types)

    if text_mask.sum() == 0:
        logger.info("No events require text generation")
        return df

    logger.info(f"Generating text for {text_mask.sum():,} events (out of {len(df):,})")

    chunk_size = 50000
    text_indices = df[text_mask].index

    for i in range(0, len(text_indices), chunk_size):
        chunk_indices = text_indices[i:i + chunk_size]
        chunk = df.loc[chunk_indices].copy()

        chunk["message_text"] = chunk.apply(
            lambda row: _generate_text(row, templates),
            axis=1
        )

        chunk["message_length"] = chunk["message_text"].str.len().fillna(0).astype(int)
        chunk["contains_url"] = chunk["message_text"].str.contains(
            r"http|www\.", case=False, na=False
        )
        chunk["contains_phone"] = chunk["message_text"].str.contains(
            r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", na=False
        )
        chunk["num_special_chars"] = chunk["message_text"].apply(
            lambda x: sum(c in "!@
        )

        df.loc[chunk_indices, "message_text"] = chunk["message_text"]
        df.loc[chunk_indices, "message_length"] = chunk["message_length"]
        df.loc[chunk_indices, "contains_url"] = chunk["contains_url"]
        df.loc[chunk_indices, "contains_phone"] = chunk["contains_phone"]
        df.loc[chunk_indices, "num_special_chars"] = chunk["num_special_chars"]

        if (i + chunk_size) % 100000 == 0:
            logger.info(f"Processed {i + chunk_size:,} / {len(text_indices):,} text events")

    logger.info("Text/Message content features (1.9) added")
    return df


def _generate_text(row: pd.Series, templates: dict) -> str:
    
    event_type = row.get("event_type", "transaction")
    is_fraud = row.get("is_fraud", False)

    if is_fraud:
        category = "fraudulent"
    else:
        category = "legitimate"

    category_templates = templates.get(category, {})
    event_templates = category_templates.get(event_type, [])

    if not event_templates:
        return f"{event_type.capitalize()} of ${row.get('amount', 0):.2f}"

    template = random.choice(event_templates)

    try:
        return template.format(
            amount=row.get("amount", 0),
            currency=row.get("currency", "USD"),
            recipient=row.get("recipient_account_id", "unknown"),
            channel=row.get("channel", "web")
        )
    except (KeyError, ValueError):
        return template