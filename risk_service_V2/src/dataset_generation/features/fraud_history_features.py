from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("fraud_history_features")


def compute_fraud_probability_vectorized(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Calcula probabilidade de fraude de forma vetorizada.
    ✅ CORRIGIDO: Agora respeita a taxa de fraude configurada!
    """
    # ✅ USAR A TAXA CORRETA DO CONFIG
    target_fraud_rate = config.get("fraud", {}).get("global_fraud_rate", 0.138)
    logger.info(f"🎯 Target fraud rate: {target_fraud_rate:.4f} ({target_fraud_rate * 100:.2f}%)")

    # Multiplicadores
    multipliers = config.get("fraud_multipliers", {})
    temporal = config.get("temporal_patterns", {})

    # 1. Por classe de risco do usuário
    user_risk_map = multipliers.get("by_user_risk_class", {
        "low": 0.3,
        "medium": 1.0,
        "high": 3.0,
        "very_high": 5.0
    })
    user_mult = df["user_risk_class"].map(user_risk_map).fillna(1.0)

    # 2. Por canal
    channel_map = multipliers.get("by_channel", {
        "web": 1.0,
        "mobile_android": 0.8,
        "mobile_ios": 0.7,
        "atm": 0.5,
        "call_center": 1.5,
        "api_partner": 2.0
    })
    channel_mult = df["channel"].map(channel_map).fillna(1.0)

    # 3. Por tipo de transação (só para transactions)
    tx_type_map = multipliers.get("by_transaction_type", {
        "pix": 1.2,
        "internal_transfer": 0.8,
        "bill_payment": 0.5,
        "card": 1.0,
        "wire": 1.3
    })
    tx_mult = df["transaction_type"].map(tx_type_map).fillna(1.0)

    # 4. Por hora do dia
    hour = pd.to_datetime(df["timestamp_utc"]).dt.hour

    hour_mult_map = temporal.get("fraud_hour_multiplier", {
        "night": 1.8,
        "business": 1.0,
        "evening": 1.3
    })

    hour_mult = pd.Series(1.0, index=df.index)
    hour_mult[hour < 6] = hour_mult_map.get("night", 1.8)
    hour_mult[(hour >= 6) & (hour < 19)] = hour_mult_map.get("business", 1.0)
    hour_mult[hour >= 19] = hour_mult_map.get("evening", 1.3)

    # 5. Por dia da semana
    dow = pd.to_datetime(df["timestamp_utc"]).dt.dayofweek

    dow_mult_map = temporal.get("fraud_day_multiplier", {
        "weekday": 1.0,
        "weekend": 1.4
    })

    dow_mult = pd.Series(dow_mult_map.get("weekday", 1.0), index=df.index)
    dow_mult[dow >= 5] = dow_mult_map.get("weekend", 1.4)

    # ✅ CALCULAR MULTIPLICADOR COMBINADO
    combined_mult = user_mult * channel_mult * tx_mult * hour_mult * dow_mult

    # ✅ NORMALIZAR para que a média seja igual à taxa alvo
    mean_mult = combined_mult.mean()
    logger.info(f"📊 Mean combined multiplier: {mean_mult:.4f}")

    # Normalizar multiplicadores
    normalized_mult = combined_mult / mean_mult

    # Calcular probabilidade final
    prob = target_fraud_rate * normalized_mult

    # Aplicar limite máximo
    max_prob = multipliers.get("max_fraud_probability", 0.40)
    prob = prob.clip(upper=max_prob)

    # ✅ VERIFICAR SE A TAXA FINAL ESTÁ CORRETA
    expected_fraud_count = int(len(df) * target_fraud_rate)
    actual_mean_prob = prob.mean()

    logger.info(f"📈 Fraud probability stats:")
    logger.info(f"  Target rate: {target_fraud_rate:.4f} ({target_fraud_rate * 100:.2f}%)")
    logger.info(f"  Actual mean prob: {actual_mean_prob:.4f} ({actual_mean_prob * 100:.2f}%)")
    logger.info(f"  Expected frauds: ~{expected_fraud_count:,}")
    logger.info(f"  Min: {prob.min():.4f}, Max: {prob.max():.4f}")
    logger.info(f"  Median: {prob.median():.4f}, Std: {prob.std():.4f}")

    return prob


def add_fraud_history_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.8 Fraud / Abuse / Historical Risk features."""

    df = events.copy()

    # Carregar configuração
    config = default_config_loader.load("dataset_config.yaml")

    # Initialize all fraud history columns with safe defaults
    df["previous_fraud_count"] = 0
    df["previous_chargeback_count"] = 0
    df["account_takeover_flag"] = 0
    df["velocity_alert_flag"] = 0
    df["blacklist_hit"] = 0
    df["whitelist_hit"] = 0
    df["money_mule_score"] = 0.0
    df["device_fingerprint_match_count"] = 0
    df["ip_reputation_score"] = np.random.uniform(0.3, 0.9, size=len(df))

    # ✅ Calcular probabilidade de fraude (CORRIGIDO)
    logger.info(f"Computing fraud probability for {len(df):,} events...")
    df["fraud_probability"] = compute_fraud_probability_vectorized(df, config)

    # ✅ Gerar flag is_fraud baseado na probabilidade
    random_values = np.random.rand(len(df))
    df["is_fraud"] = (random_values < df["fraud_probability"]).astype(int)

    fraud_count = df["is_fraud"].sum()
    fraud_rate = fraud_count / len(df) * 100

    logger.info("=" * 80)
    logger.info(f"🎯 FRAUD GENERATION RESULTS:")
    logger.info(f"  Total events: {len(df):,}")
    logger.info(f"  Fraudulent: {fraud_count:,} ({fraud_rate:.2f}%)")
    logger.info(f"  Legitimate: {len(df) - fraud_count:,} ({100 - fraud_rate:.2f}%)")
    logger.info("=" * 80)

    # ✅ Atribuir fraud_type baseado em cenários
    fraud_scenarios = config.get("fraud", {}).get("scenario_weights", {})
    scenario_names = list(fraud_scenarios.keys())
    scenario_weights = list(fraud_scenarios.values())

    # Normalizar pesos
    total_weight = sum(scenario_weights)
    scenario_probs = [w / total_weight for w in scenario_weights]

    fraud_mask = df["is_fraud"] == 1
    num_frauds = fraud_mask.sum()

    if num_frauds > 0:
        # Atribuir tipos de fraude aleatoriamente baseado nos pesos
        fraud_types = np.random.choice(
            scenario_names,
            size=num_frauds,
            p=scenario_probs
        )
        df.loc[fraud_mask, "fraud_type"] = fraud_types

        # Log distribuição de tipos
        logger.info("📊 Fraud type distribution:")
        for scenario in scenario_names:
            count = (df["fraud_type"] == scenario).sum()
            pct = count / num_frauds * 100 if num_frauds > 0 else 0
            logger.info(f"  • {scenario:30s}: {count:>6,} ({pct:>5.1f}%)")

    # Money mule detection
    if "recipient_id" in df.columns:
        unique_recipients = df["recipient_id"].dropna().unique()

        if len(unique_recipients) > 0:
            num_mules = max(1, int(len(unique_recipients) * 0.01))
            mule_recipients = np.random.choice(
                unique_recipients,
                size=num_mules,
                replace=False
            )

            mask = df["recipient_id"].isin(mule_recipients)
            df.loc[mask, "money_mule_score"] = np.random.uniform(0.6, 0.95, size=mask.sum())

    # Velocity alerts
    if "transactions_last_1h" in df.columns:
        df.loc[df["transactions_last_1h"] > 10, "velocity_alert_flag"] = 1

    # Blacklist/whitelist
    df["blacklist_hit"] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
    df["whitelist_hit"] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])

    # Device fingerprint matches
    if "device_id" in df.columns:
        device_counts = df.groupby("device_id").size()
        df["device_fingerprint_match_count"] = df["device_id"].map(device_counts).fillna(1).astype(int)

    # Previous fraud/chargeback counts
    if "user_risk_class" in df.columns:
        high_risk_mask = df["user_risk_class"] == "high"
        df.loc[high_risk_mask, "previous_fraud_count"] = np.random.poisson(2, size=high_risk_mask.sum())
        df.loc[high_risk_mask, "previous_chargeback_count"] = np.random.poisson(1, size=high_risk_mask.sum())

    # Account takeover flag
    df["account_takeover_flag"] = np.random.choice([0, 1], size=len(df), p=[0.995, 0.005])

    logger.info("✅ Fraud/historical risk features (1.8) added")
    return df