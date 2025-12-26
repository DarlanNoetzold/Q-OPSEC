from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger
from src.dataset_generation.utils.ollama_client import OllamaClient

logger = get_logger("llm_features")


def add_llm_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.10 LLM-derived features.

    Fields:
      - llm_risk_score_transaction (0-1)
      - llm_risk_label_transaction (low, medium, high)
      - llm_phishing_score (0-1)
      - llm_phishing_label (legitimate, suspicious, phishing)
      - llm_explanation_short (string)

    This module uses OllamaClient to process a sample of events based on
    dataset_config.yaml settings.
    """

    df = events.copy()
    dataset_cfg = default_config_loader.load("dataset_config.yaml")
    llm_cfg = dataset_cfg.get("llm_processing", {})

    if not llm_cfg.get("enabled", False):
        logger.info("LLM processing disabled in config. Skipping.")
        return _add_empty_llm_columns(df)

    client = OllamaClient()
    if not client.check_connection():
        logger.error("Could not connect to Ollama. Skipping LLM features.")
        return _add_empty_llm_columns(df)

    sample_rate = float(llm_cfg.get("sample_rate", 0.1))

    # Initialize columns
    df = _add_empty_llm_columns(df)

    # Identify events to process
    # We prioritize transactions and events with text
    tx_mask = df["event_type"] == "transaction"
    text_mask = df["message_text"].notna()

    eligible_indices = df[tx_mask | text_mask].index.tolist()
    num_to_sample = int(len(eligible_indices) * sample_rate)

    if num_to_sample == 0:
        logger.info("No events sampled for LLM processing.")
        return df

    sampled_indices = np.random.choice(eligible_indices, size=num_to_sample, replace=False)

    logger.info("Processing {n} events with LLM (Ollama)...", n=num_to_sample)

    # Process in loop (could be batched if Ollama supported it better)
    for idx in tqdm(sampled_indices, desc="LLM Processing"):
        row = df.loc[idx]

        # 1. Transaction Risk
        if row["event_type"] == "transaction":
            # Prepare context for LLM
            tx_data = {
                "amount": row.get("amount"),
                "currency": row.get("currency"),
                "type": row.get("transaction_type"),
                "channel": row.get("channel"),
                "recipient_type": row.get("recipient_type"),
                "is_new_recipient": row.get("is_new_recipient"),
                "distance_km": row.get("distance_from_registered_location_km"),
                "speed_kmh": row.get("speed_from_last_event_kmh"),
                "is_proxy": row.get("is_proxy"),
                "historical_risk": row.get("risk_score_historical")
            }
            res = client.assess_transaction_risk(tx_data)
            df.at[idx, "llm_risk_score_transaction"] = res.get("risk_score", 0.0)
            df.at[idx, "llm_risk_label_transaction"] = res.get("risk_label", "unknown")
            df.at[idx, "llm_explanation_short"] = res.get("explanation", "")

        # 2. Phishing Detection (if text exists)
        if pd.notna(row["message_text"]):
            res = client.detect_phishing(row["message_text"])
            df.at[idx, "llm_phishing_score"] = res.get("phishing_score", 0.0)
            df.at[idx, "llm_phishing_label"] = res.get("phishing_label", "unknown")
            # Append explanation if already exists from transaction
            existing_exp = df.at[idx, "llm_explanation_short"]
            new_exp = res.get("explanation", "")
            if existing_exp:
                df.at[idx, "llm_explanation_short"] = f"{existing_exp} | {new_exp}"
            else:
                df.at[idx, "llm_explanation_short"] = new_exp

    logger.info("LLM features (1.10) added to sampled events")
    return df


def _add_empty_llm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize LLM columns with default/fallback values."""
    llm_config = default_config_loader.load("llm_config.yaml")
    fallbacks = llm_config.get("fallbacks", {})

    cols = {
        "llm_risk_score_transaction": fallbacks.get("risk_score", 0.0),
        "llm_risk_label_transaction": "not_evaluated",
        "llm_phishing_score": fallbacks.get("phishing_score", 0.0),
        "llm_phishing_label": "not_evaluated",
        "llm_explanation_short": None
    }

    for col, val in cols.items():
        if col not in df.columns:
            df[col] = val

    return df