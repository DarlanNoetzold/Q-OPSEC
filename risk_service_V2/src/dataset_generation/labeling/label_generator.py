from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger


logger = get_logger("labeling")


def apply_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.11 Labels / Ground Truth.

    Fields:
      - is_fraud (0/1)
      - fraud_type (e.g., account_takeover, card_testing, money_mule, phishing_induced, none)
      - label_source (synthetic_rule, scenario_definition, manual_override)

    Strategy (synthetic dataset):
      - Use fraud_scenarios.yaml to set target overall fraud_rate.
      - Sample a subset of events as fraud, with different patterns per scenario.
      - For now, we use simple heuristics + random sampling by event_type/amount.
    """

    df = events.copy()

    scenarios_cfg = default_config_loader.load("fraud_scenarios.yaml")
    dataset_cfg = default_config_loader.load("dataset_config.yaml")

    fraud_rate = float(dataset_cfg.get("generation", {}).get("fraud_rate_overall", 0.01))

    # Initialize labels
    df["is_fraud"] = 0
    df["fraud_type"] = "none"
    df["label_source"] = "synthetic_rule"

    n = len(df)
    target_frauds = int(n * fraud_rate)
    if target_frauds == 0:
        logger.info("Target fraud count is 0; skipping fraud labeling.")
        return df

    logger.info("Targeting approximately {n} fraudulent events (rate={rate:.4f})", n=target_frauds, rate=fraud_rate)

    # Simple scoring / prioritization for sampling
    # We prefer to mark as fraud:
    #  - high amounts
    #  - new recipients
    #  - international transactions
    #  - high historical risk
    score = np.zeros(n)

    # Only consider transactions for now
    tx_mask = df["event_type"] == "transaction"
    idx_tx = np.where(tx_mask)[0]

    if len(idx_tx) == 0:
        logger.warning("No transactions found; fraud labels will be all 0.")
        return df

    # Build score for transactions
    score[tx_mask] += (df.loc[tx_mask, "amount"].fillna(0.0) / (df.loc[tx_mask, "amount"].fillna(0.0).median() + 1e-6)).clip(0, 10)
    score[tx_mask] += df.loc[tx_mask, "is_new_recipient"].fillna(0) * 2.0
    if "is_international" in df.columns:
        score[tx_mask] += df.loc[tx_mask, "is_international"].fillna(0) * 2.0
    if "risk_score_historical" in df.columns:
        score[tx_mask] += df.loc[tx_mask, "risk_score_historical"].fillna(0) * 3.0
    if "llm_risk_score_transaction" in df.columns:
        score[tx_mask] += df.loc[tx_mask, "llm_risk_score_transaction"].fillna(0) * 2.0

    # Normalize scores and sample top K as fraud
    score_tx = score[idx_tx]
    if score_tx.sum() <= 0:
        # fallback: uniform random among transactions
        fraud_indices_tx = np.random.choice(idx_tx, size=min(target_frauds, len(idx_tx)), replace=False)
    else:
        probs = score_tx / score_tx.sum()
        fraud_indices_tx = np.random.choice(idx_tx, size=min(target_frauds, len(idx_tx)), replace=False, p=probs)

    df.loc[fraud_indices_tx, "is_fraud"] = 1

    # Assign fraud_type roughly based on scenarios and simple heuristics
    scenario_weights = scenarios_cfg.get("scenario_weights", {
        "account_takeover": 0.3,
        "card_testing": 0.2,
        "money_mule": 0.2,
        "phishing_induced": 0.3,
    })
    types, weights = zip(*scenario_weights.items())

    # Assign type per fraudulent event
    for idx in fraud_indices_tx:
        row = df.loc[idx]
        ftype = random.choices(types, weights=weights, k=1)[0]

        # Optionally refine type by characteristics
        if row.get("is_new_recipient", 0) == 1 and row.get("amount", 0) > df["amount"].median():
            # likely mule or phishing induced
            ftype = random.choice(["money_mule", "phishing_induced"])
        elif row.get("llm_phishing_score", 0) > 0.7:
            ftype = "phishing_induced"
        elif row.get("transactions_last_1h", 0) > 3 and row.get("amount", 0) < df["amount"].median():
            ftype = "card_testing"

        df.at[idx, "fraud_type"] = ftype
        df.at[idx, "label_source"] = "scenario_definition"

    logger.info("Applied fraud labels to {n} events", n=int(df["is_fraud"].sum()))
    return df