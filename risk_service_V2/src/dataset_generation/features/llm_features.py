from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger
from src.dataset_generation.utils.ollama_client import OllamaClient

logger = get_logger("llm_features")


def add_llm_features(events: pd.DataFrame) -> pd.DataFrame:
    

    df = events.copy()

    dataset_cfg = default_config_loader.load("dataset_config.yaml")
    llm_cfg = dataset_cfg.get("llm", {})

    llm_enabled = llm_cfg.get("enabled", False)

    if not llm_enabled:
        logger.info("LLM processing disabled in config. Skipping.")
        df["llm_risk_score"] = 0.0
        df["llm_risk_reasoning"] = None
        df["llm_phishing_detected"] = 0
        df["llm_sentiment_score"] = 0.0
        df["llm_urgency_score"] = 0.0
        return df

    try:
        client = OllamaClient()
        if not client.test_connection():
            logger.warning("Ollama not available. Skipping LLM features.")
            df["llm_risk_score"] = 0.0
            df["llm_risk_reasoning"] = None
            df["llm_phishing_detected"] = 0
            df["llm_sentiment_score"] = 0.0
            df["llm_urgency_score"] = 0.0
            return df
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        df["llm_risk_score"] = 0.0
        df["llm_risk_reasoning"] = None
        df["llm_phishing_detected"] = 0
        df["llm_sentiment_score"] = 0.0
        df["llm_urgency_score"] = 0.0
        return df

    sample_rates = llm_cfg.get("sample_rate", {})
    batch_size = llm_cfg.get("batch_size", 16)

    df["llm_risk_score"] = 0.0
    df["llm_risk_reasoning"] = None
    df["llm_phishing_detected"] = 0
    df["llm_sentiment_score"] = 0.0
    df["llm_urgency_score"] = 0.0

    events_to_process = []

    for event_type, rate in sample_rates.items():
        if event_type == "other":
            mask = ~df["event_type"].isin(["transaction", "login"])
        else:
            mask = df["event_type"] == event_type

        type_events = df[mask]
        if len(type_events) > 0:
            sample_size = int(len(type_events) * rate)
            if sample_size > 0:
                sampled = type_events.sample(n=min(sample_size, len(type_events)), random_state=42)
                events_to_process.extend(sampled.index.tolist())

    logger.info(f"Processing {len(events_to_process)} events with LLM (out of {len(df)} total)")

    for i in range(0, len(events_to_process), batch_size):
        batch_indices = events_to_process[i:i + batch_size]

        for idx in batch_indices:
            row = df.loc[idx]

            context = {
                "event_type": row.get("event_type"),
                "amount": row.get("amount"),
                "channel": row.get("channel"),
                "message_text": row.get("message_text"),
                "user_risk_class": row.get("user_risk_class"),
            }

            try:
                risk_result = client.assess_transaction_risk(context)
                df.loc[idx, "llm_risk_score"] = risk_result.get("risk_score", 0.0)
                df.loc[idx, "llm_risk_reasoning"] = risk_result.get("reasoning", "")
            except Exception as e:
                logger.warning(f"LLM risk assessment failed for event {idx}: {e}")

            if pd.notna(row.get("message_text")):
                try:
                    phishing_result = client.detect_phishing(row["message_text"])
                    df.loc[idx, "llm_phishing_detected"] = 1 if phishing_result.get("is_phishing", False) else 0
                    df.loc[idx, "llm_sentiment_score"] = phishing_result.get("sentiment_score", 0.0)
                    df.loc[idx, "llm_urgency_score"] = phishing_result.get("urgency_score", 0.0)
                except Exception as e:
                    logger.warning(f"LLM phishing detection failed for event {idx}: {e}")

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch_indices)}/{len(events_to_process)} events")

    logger.info("LLM features added")
    return df