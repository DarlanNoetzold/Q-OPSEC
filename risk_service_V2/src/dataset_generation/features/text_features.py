from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("text_message_features")


def add_text_message_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.9 Text / Message Content features.

    Fields:
      - message_text (raw text)
      - message_length_chars
      - message_length_tokens (approximate)
      - has_url (0/1)
      - has_phone_number (0/1)
      - has_suspicious_keywords (0/1)
      - detected_language (e.g., pt, en)
      - message_category (e.g., notification, transfer_memo, chat)

    Note: This module generates synthetic text based on event_type and
    fraud scenarios (using templates from fraud_scenarios.yaml).
    """

    df = events.copy()
    scenarios_cfg = default_config_loader.load("fraud_scenarios.yaml")

    # Templates
    legit_templates = scenarios_cfg.get("legitimate_templates", ["Payment for services", "Transfer to friend"])
    phishing_templates = scenarios_cfg.get("fraud_scenarios", {}).get("phishing_induced", {}).get("text_templates", [
        "Urgent: verify your account"])

    # Initialize columns
    df["message_text"] = None
    df["message_length_chars"] = 0
    df["message_length_tokens"] = 0
    df["has_url"] = 0
    df["has_phone_number"] = 0
    df["has_suspicious_keywords"] = 0
    df["detected_language"] = "pt"  # Default
    df["message_category"] = "unknown"

    # We only generate text for certain event types (e.g., transaction, message)
    # For this simulation, let's assume transactions and some 'message' events have text.
    text_mask = df["event_type"].isin(["transaction", "message"])

    # In a real pipeline, we'd know if it's fraud or not.
    # Here we can use a placeholder or wait for the labeling module.
    # Let's assume a small % are suspicious for now.

    def _generate_text(row):
        if row["event_type"] not in ["transaction", "message"]:
            return None

        # Simple heuristic: if it's a high risk user or specific scenario, use phishing template
        is_suspicious = False
        if "user_risk_class" in row and row["user_risk_class"] == "high":
            is_suspicious = random.random() < 0.3

        if is_suspicious:
            text = random.choice(phishing_templates)
        else:
            text = random.choice(legit_templates)

        # Add some randomness/personalization
        if "{amount}" in text and "amount" in row:
            text = text.replace("{amount}", str(row["amount"]))
        if "{recipient_id}" in text and "recipient_id" in row:
            text = text.replace("{recipient_id}", str(row["recipient_id"]))

        return text

    df.loc[text_mask, "message_text"] = df[text_mask].apply(_generate_text, axis=1)

    # Compute derived features
    valid_text_mask = df["message_text"].notna()

    df.loc[valid_text_mask, "message_length_chars"] = df.loc[valid_text_mask, "message_text"].str.len()
    df.loc[valid_text_mask, "message_length_tokens"] = (df.loc[valid_text_mask, "message_length_chars"] / 4).astype(
        int)  # Rough approx

    # Simple regex-like checks
    suspicious_keywords = ["urgente", "bloqueio", "verificar", "senha", "promoção", "ganhou", "clique"]

    def _check_content(text):
        if not text: return 0, 0, 0
        has_url = 1 if ("http" in text or "www" in text or ".com" in text) else 0
        has_phone = 1 if any(c.isdigit() for c in text) and len([c for c in text if c.isdigit()]) >= 8 else 0
        has_suspicious = 1 if any(k in text.lower() for k in suspicious_keywords) else 0
        return has_url, has_phone, has_suspicious

    content_results = df.loc[valid_text_mask, "message_text"].apply(_check_content)
    if not content_results.empty:
        df.loc[valid_text_mask, "has_url"] = [r[0] for r in content_results]
        df.loc[valid_text_mask, "has_phone_number"] = [r[1] for r in content_results]
        df.loc[valid_text_mask, "has_suspicious_keywords"] = [r[2] for r in content_results]

    # Category
    df.loc[df["event_type"] == "transaction", "message_category"] = "transfer_memo"
    df.loc[df["event_type"] == "message", "message_category"] = "chat"

    logger.info("Text/Message content features (1.9) added")
    return df