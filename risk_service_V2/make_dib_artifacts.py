#!/usr/bin/env python3
"""
Script completo para gerar dataset sintético de detecção de fraude
com TODAS as colunas necessárias + figuras estatísticas avançadas + tabelas LaTeX.

Uso:
    python generate_complete_dataset_and_figures_v3.py --output-dir output_v3 --num-users 50000 --year 2025
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

USER_PROFILES = {
    "retail_low_risk": {
        "weight": 0.60,
        "events_per_month_mean": 25,
        "events_per_month_std": 8,
        "amount_mean": 180.0,
        "amount_cv": 1.1,
        "amount_cap": 50000.0,
        "user_type": "individual",
        "user_segment": "retail",
        "user_risk_class": "low",
        "channel_dist": {"mobile_android": 0.45, "mobile_ios": 0.35, "web": 0.20},
        "tx_type_dist": {"pix": 0.50, "card": 0.30, "bill_payment": 0.15, "internal_transfer": 0.05},
    },
    "retail_medium_risk": {
        "weight": 0.25,
        "events_per_month_mean": 35,
        "events_per_month_std": 12,
        "amount_mean": 420.0,
        "amount_cv": 1.3,
        "amount_cap": 100000.0,
        "user_type": "individual",
        "user_segment": "retail",
        "user_risk_class": "medium",
        "channel_dist": {"mobile_android": 0.40, "mobile_ios": 0.25, "web": 0.30, "atm": 0.05},
        "tx_type_dist": {"pix": 0.45, "card": 0.25, "wire": 0.15, "internal_transfer": 0.15},
    },
    "business": {
        "weight": 0.10,
        "events_per_month_mean": 80,
        "events_per_month_std": 25,
        "amount_mean": 2500.0,
        "amount_cv": 1.5,
        "amount_cap": 500000.0,
        "user_type": "business",
        "user_segment": "business",
        "user_risk_class": "medium",
        "channel_dist": {"web": 0.60, "api_partner": 0.25, "mobile_android": 0.10, "mobile_ios": 0.05},
        "tx_type_dist": {"wire": 0.40, "pix": 0.30, "internal_transfer": 0.20, "bill_payment": 0.10},
    },
    "high_risk": {
        "weight": 0.05,
        "events_per_month_mean": 50,
        "events_per_month_std": 20,
        "amount_mean": 800.0,
        "amount_cv": 1.8,
        "amount_cap": 150000.0,
        "user_type": "individual",
        "user_segment": "retail",
        "user_risk_class": "high",
        "channel_dist": {"mobile_android": 0.35, "web": 0.30, "call_center": 0.20, "atm": 0.15},
        "tx_type_dist": {"pix": 0.40, "card": 0.30, "wire": 0.20, "internal_transfer": 0.10},
    },
}

EVENT_MIX = {
    "transaction": 0.75,
    "login": 0.18,
    "password_change": 0.02,
    "email_change": 0.01,
    "phone_change": 0.01,
    "address_change": 0.01,
    "mfa_setup": 0.01,
    "device_registration": 0.01,
}

FRAUD_SCENARIOS = {
    "account_takeover": 0.25,
    "card_fraud": 0.20,
    "synthetic_identity": 0.15,
    "money_laundering": 0.15,
    "phishing": 0.10,
    "insider_fraud": 0.08,
    "refund_fraud": 0.07,
}

FRAUD_MULTIPLIERS = {
    "by_user_risk_class": {"low": 0.3, "medium": 1.0, "high": 3.5, "very_high": 6.0},
    "by_channel": {
        "mobile_android": 0.9,
        "mobile_ios": 0.7,
        "web": 1.0,
        "atm": 0.5,
        "call_center": 1.6,
        "api_partner": 2.2,
    },
    "by_transaction_type": {
        "pix": 1.3,
        "wire": 1.4,
        "card": 1.0,
        "internal_transfer": 0.7,
        "bill_payment": 0.4,
    },
    "by_hour": {
        "night": 2.0,
        "business": 1.0,
        "evening": 1.4,
    },
    "by_dow": {"weekday": 1.0, "weekend": 1.5},
}

TARGET_FRAUD_RATE = 0.138
MAX_FRAUD_PROBABILITY = 0.45

COUNTRIES = ["BR", "US", "MX", "AR", "CO", "CL", "PE"]
REGIONS_BR = ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Bahia", "Paraná", "Rio Grande do Sul"]
TIMEZONES = ["America/Sao_Paulo", "America/New_York", "America/Mexico_City", "America/Argentina/Buenos_Aires"]

REQUIRED_COLUMNS = [
    "event_id", "user_id", "account_id", "event_type", "event_source", "timestamp_utc", "timezone",
    "amount", "currency", "transaction_type", "channel", "timestamp_local", "user_type", "user_segment",
    "user_risk_class", "registered_country", "registered_region", "account_creation_date", "account_age_days",
    "sensitive_data_change_last_7d", "registered_devices_count", "active_devices_last_30d", "hour_of_day",
    "day_of_week", "is_weekend", "is_local_holiday", "seconds_since_last_login", "seconds_since_last_transaction",
    "transactions_last_1h", "transactions_last_24h", "transactions_last_7d", "transactions_last_30d",
    "amount_sum_last_24h", "amount_sum_last_7d", "amount_sum_last_30d", "amount_mean_last_30d", "amount_std_last_30d",
    "logins_last_24h", "login_failures_last_24h", "password_resets_last_30d", "transaction_description",
    "beneficiary_bank_code", "merchant_category_code", "is_round_amount", "is_international",
    "amount_increase_vs_30d_mean",
    "ip_address", "ip_country", "ip_region", "ip_city", "ip_asn", "ip_isp", "is_vpn", "is_datacenter_ip",
    "geolocation_lat", "geolocation_lon", "ip_blacklisted", "is_proxy", "is_tor", "connection_type",
    "is_location_anomaly",
    "distance_from_registered_location_km", "device_id", "device_type", "os_family", "os_version", "browser_family",
    "browser_version", "app_version", "is_emulator", "is_rooted_or_jailbroken", "is_new_device_for_user",
    "devices_last_30d",
    "is_device_compromised", "tls_version", "cipher_strength_bits", "waf_present", "waf_vendor", "ids_present",
    "ips_present",
    "antimalware_present", "antimalware_vendor", "mfa_enabled", "mfa_method", "device_binding_enabled",
    "risk_engine_version",
    "security_score_tech", "previous_fraud_count", "previous_chargeback_count", "account_takeover_flag",
    "velocity_alert_flag",
    "blacklist_hit", "whitelist_hit", "money_mule_score", "device_fingerprint_match_count", "ip_reputation_score",
    "fraud_probability", "is_fraud", "fraud_type", "message_text", "message_length", "message_language", "contains_url",
    "contains_phone", "num_special_chars", "llm_risk_score", "llm_risk_reasoning", "llm_phishing_detected",
    "llm_sentiment_score", "llm_urgency_score",
]


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def make_versioned_output_dir(base_dir: Path) -> Path:
    """Cria diretório de saída versionado: base_dir, base_dir_001, base_dir_002, ..."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir
    i = 1
    while True:
        cand = Path(f"{str(base_dir)}_{i:03d}")
        if not cand.exists():
            return cand
        i += 1


def to_latex_table(df: pd.DataFrame, caption: str, label: str, index: bool = False) -> str:
    """Gera tabela LaTeX com formatação simples (booktabs)."""
    return df.to_latex(
        index=index,
        escape=True,
        longtable=False,
        caption=caption,
        label=label,
        bold_rows=False,
        na_rep="NA",
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        column_format=None,
        multicolumn=True,
        multicolumn_format="c",
    )


def stable_hash(s: str) -> int:
    """Hash estável para reprodutibilidade."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def lognormal_amount(mean: float, cv: float) -> float:
    """Gera valor lognormal com média e coeficiente de variação."""
    cv = max(0.05, float(cv))
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(max(1e-6, mean)) - 0.5 * sigma2
    return float(np.random.lognormal(mean=mu, sigma=sigma))


def month_factor(month: int) -> float:
    """Sazonalidade mensal (fim de ano maior)."""
    factors = {1: 0.88, 2: 0.92, 3: 0.96, 4: 0.98, 5: 1.00, 6: 1.02,
               7: 1.01, 8: 1.03, 9: 1.04, 10: 1.06, 11: 1.10, 12: 1.18}
    return factors.get(int(month), 1.0)


def dow_factor(dow: int) -> float:
    """Fator por dia da semana (fim de semana maior)."""
    return 1.12 if int(dow) >= 5 else 1.00


def hour_factor(hour: int) -> float:
    """Fator por hora do dia."""
    hour = int(hour)
    if 8 <= hour <= 18:
        return 1.05
    if 19 <= hour <= 23:
        return 1.08
    return 0.92


@dataclass
class DeviceProfile:
    device_id: str
    device_type: str
    os_family: str
    os_version: str
    browser_family: str
    browser_version: str
    app_version: str
    is_emulator: int
    is_rooted_or_jailbroken: int


def rand_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    xs = [x for x, _ in items]
    ws = [w for _, w in items]
    return rng.choices(xs, weights=ws, k=1)[0]


def make_device(rng: random.Random, user_id: str, idx: int, channel_hint: str) -> DeviceProfile:
    """Cria perfil de device realista baseado no canal."""
    if channel_hint in {"mobile_android", "mobile_ios"}:
        device_type = rand_choice(rng, [("mobile", 0.92), ("tablet", 0.08)])
        if channel_hint == "mobile_android":
            os_family = "Android"
            os_version = rand_choice(rng, [("10", 0.10), ("11", 0.20), ("12", 0.25), ("13", 0.25), ("14", 0.20)])
        else:
            os_family = "iOS"
            os_version = rand_choice(rng, [("15", 0.20), ("16", 0.45), ("17", 0.35)])
        browser_family = rand_choice(rng, [("in_app", 0.70), ("Chrome", 0.20), ("Safari", 0.10)])
        browser_version = str(rng.randint(90, 125))
        app_version = f"{rng.randint(4, 8)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
    else:
        device_type = rand_choice(rng, [("desktop", 0.82), ("mobile", 0.12), ("tablet", 0.06)])
        os_family = rand_choice(rng, [("Windows", 0.62), ("macOS", 0.25), ("Linux", 0.13)])
        os_version = (
            rand_choice(rng, [("10", 0.25), ("11", 0.75)]) if os_family == "Windows"
            else rand_choice(rng, [("12", 0.20), ("13", 0.35), ("14", 0.30), ("15", 0.15)]) if os_family == "macOS"
            else rand_choice(rng, [("Ubuntu22", 0.45), ("Ubuntu24", 0.30), ("Other", 0.25)])
        )
        browser_family = rand_choice(rng, [("Chrome", 0.62), ("Edge", 0.18), ("Firefox", 0.12), ("Safari", 0.08)])
        browser_version = str(rng.randint(90, 125))
        app_version = None

    is_emulator = 1 if rng.random() < (0.008 if channel_hint in {"mobile_android", "mobile_ios"} else 0.001) else 0
    is_rooted = 1 if rng.random() < (0.012 if channel_hint == "mobile_android" else 0.004) else 0

    device_id = f"D{stable_hash(f'{user_id}:{idx}:{os_family}:{device_type}') % 10_000_000:07d}"
    return DeviceProfile(
        device_id=device_id,
        device_type=device_type,
        os_family=os_family,
        os_version=os_version,
        browser_family=browser_family,
        browser_version=browser_version,
        app_version=app_version,
        is_emulator=is_emulator,
        is_rooted_or_jailbroken=is_rooted,
    )


# ============================================================================
# GERAÇÃO DE USUÁRIOS
# ============================================================================

def generate_users(num_users: int, start_date: datetime, seed: int = 42) -> pd.DataFrame:
    """Gera DataFrame de usuários."""
    print(f"\n[1/6] Gerando {num_users:,} usuários...")
    random.seed(seed)
    np.random.seed(seed)

    profiles = list(USER_PROFILES.keys())
    weights = [USER_PROFILES[p]["weight"] for p in profiles]

    records = []
    for i in range(num_users):
        profile_name = random.choices(profiles, weights=weights, k=1)[0]
        profile = USER_PROFILES[profile_name]

        days_before = random.randint(0, 3 * 365)
        account_creation = start_date - timedelta(days=days_before)

        records.append({
            "user_id": f"U{i:07d}",
            "account_id": f"A{i:07d}",
            "profile_name": profile_name,
            "user_type": profile["user_type"],
            "user_segment": profile["user_segment"],
            "user_risk_class": profile["user_risk_class"],
            "registered_country": random.choices(COUNTRIES, weights=[0.70, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03], k=1)[0],
            "registered_region": random.choice(REGIONS_BR),
            "timezone": random.choice(TIMEZONES),
            "account_creation_date": account_creation,
        })

    df = pd.DataFrame.from_records(records)
    df["account_creation_date"] = pd.to_datetime(df["account_creation_date"]).dt.tz_localize(None)
    print(f"✅ {len(df):,} usuários gerados")
    return df


# ============================================================================
# GERAÇÃO DE EVENTOS
# ============================================================================

def generate_events(users_df: pd.DataFrame, start_date: datetime, end_date: datetime, seed: int = 42) -> pd.DataFrame:
    """Gera eventos brutos para cada usuário."""
    print(f"\n[2/6] Gerando eventos de {start_date.date()} até {end_date.date()}...")
    random.seed(seed)
    np.random.seed(seed)

    total_months = max(1.0, (end_date - start_date).days / 30.0)

    event_types = list(EVENT_MIX.keys())
    event_weights = list(EVENT_MIX.values())
    total_weight = sum(event_weights)
    event_weights = [w / total_weight for w in event_weights]

    hour_weights = np.array([
        0.5, 0.4, 0.3, 0.3, 0.4, 0.6,
        1.0, 1.5, 2.0, 2.5, 2.8, 3.0,
        3.2, 3.0, 2.8, 2.5, 2.3, 2.0,
        2.2, 2.5, 2.3, 2.0, 1.5, 1.0,
    ])
    hour_weights = hour_weights / hour_weights.sum()

    records = []
    event_counter = 0

    for _, user in users_df.iterrows():
        profile = USER_PROFILES[user["profile_name"]]

        mean_events = profile["events_per_month_mean"] * total_months
        std_events = profile["events_per_month_std"] * (total_months ** 0.5)
        n_events = max(1, int(np.random.normal(mean_events, std_events)))

        for _ in range(n_events):
            event_type = random.choices(event_types, weights=event_weights, k=1)[0]

            total_days = max(0, (end_date - start_date).days)
            base_date = start_date + timedelta(days=random.randint(0, total_days))
            hour = int(np.random.choice(np.arange(24), p=hour_weights))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts_utc = base_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                hours=hour, minutes=minute, seconds=second
            )

            channel_dist = profile["channel_dist"]
            channels = list(channel_dist.keys())
            ch_weights = list(channel_dist.values())
            channel = random.choices(channels, weights=ch_weights, k=1)[0]

            if channel in {"mobile_android", "mobile_ios"}:
                event_source = "mobile_app"
            elif channel == "web":
                event_source = "web_app"
            elif channel == "atm":
                event_source = "atm"
            elif channel == "call_center":
                event_source = "call_center"
            else:
                event_source = "api_partner"

            base_record = {
                "event_id": f"E{event_counter:010d}",
                "user_id": user["user_id"],
                "account_id": user["account_id"],
                "event_type": event_type,
                "event_source": event_source,
                "timestamp_utc": ts_utc,
                "timezone": user["timezone"],
            }

            if event_type == "transaction":
                tx_type_dist = profile["tx_type_dist"]
                tx_types = list(tx_type_dist.keys())
                tx_weights = list(tx_type_dist.values())
                transaction_type = random.choices(tx_types, weights=tx_weights, k=1)[0]

                base_mean = profile["amount_mean"]
                base_cv = profile["amount_cv"]
                amt = lognormal_amount(mean=base_mean, cv=base_cv)

                amt *= month_factor(ts_utc.month) * dow_factor(ts_utc.weekday()) * hour_factor(ts_utc.hour)
                amt = max(1.0, min(amt, profile["amount_cap"]))

                if amt < 50:
                    amt = round(amt / 5) * 5
                elif amt < 500:
                    amt = round(amt / 10) * 10
                elif amt < 5000:
                    amt = round(amt / 50) * 50
                else:
                    amt = round(amt / 100) * 100

                base_record.update({
                    "amount": float(amt),
                    "currency": "BRL",
                    "transaction_type": transaction_type,
                    "channel": channel,
                })
            else:
                base_record.update({
                    "amount": np.nan,
                    "currency": None,
                    "transaction_type": None,
                    "channel": channel,
                })

            records.append(base_record)
            event_counter += 1

    df = pd.DataFrame.from_records(records)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
    df = df.sort_values(["user_id", "timestamp_utc"]).reset_index(drop=True)

    print(f"✅ {len(df):,} eventos gerados")
    return df


# ============================================================================
# ADIÇÃO DE FEATURES
# ============================================================================

def add_all_features(events_df: pd.DataFrame, users_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Adiciona todas as features necessárias."""
    print("\n[3/6] Adicionando features...")
    random.seed(seed)
    np.random.seed(seed)

    df = events_df.copy()

    print("  → Event identification...")
    df["timestamp_local"] = df["timestamp_utc"]

    print("  → User/account features...")
    users_meta = users_df[[
        "user_id", "user_type", "user_segment", "user_risk_class",
        "registered_country", "registered_region", "account_creation_date"
    ]]
    df = df.merge(users_meta, on="user_id", how="left")

    ts_utc = pd.to_datetime(df["timestamp_utc"], errors="coerce").dt.tz_localize(None)
    acc_date = pd.to_datetime(df["account_creation_date"], errors="coerce").dt.tz_localize(None)

    df["account_age_days"] = (
        (ts_utc.dt.floor("D") - acc_date.dt.floor("D"))
        .dt.days
        .clip(lower=0)
    )

    print("  → Temporal/behavioral features...")
    df["hour_of_day"] = pd.to_datetime(df["timestamp_utc"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp_utc"]).dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_local_holiday"] = 0

    def add_temporal_per_user(g):
        g = g.sort_values("timestamp_utc").copy()
        times = pd.to_datetime(g["timestamp_utc"]).values.astype("datetime64[s]").astype("int64")
        amounts = g["amount"].fillna(0.0).values
        is_tx = (g["event_type"] == "transaction").values
        is_login = (g["event_type"] == "login").values

        g["seconds_since_last_login"] = np.nan
        g["seconds_since_last_transaction"] = np.nan
        g["transactions_last_1h"] = 0
        g["transactions_last_24h"] = 0
        g["transactions_last_7d"] = 0
        g["transactions_last_30d"] = 0
        g["amount_sum_last_24h"] = 0.0
        g["amount_sum_last_7d"] = 0.0
        g["amount_sum_last_30d"] = 0.0
        g["amount_mean_last_30d"] = 0.0
        g["amount_std_last_30d"] = 0.0
        g["logins_last_24h"] = 0
        g["login_failures_last_24h"] = 0
        g["password_resets_last_30d"] = 0

        for i in range(len(g)):
            t = times[i]
            prev = slice(0, i)

            if i > 0:
                prev_logins = times[prev][is_login[prev]]
                if len(prev_logins) > 0:
                    g.iloc[i, g.columns.get_loc("seconds_since_last_login")] = t - prev_logins[-1]

                prev_txs = times[prev][is_tx[prev]]
                if len(prev_txs) > 0:
                    g.iloc[i, g.columns.get_loc("seconds_since_last_transaction")] = t - prev_txs[-1]

            mask_1h = (times[prev] >= t - 3600) & (times[prev] < t)
            mask_24h = (times[prev] >= t - 86400) & (times[prev] < t)
            mask_7d = (times[prev] >= t - 7 * 86400) & (times[prev] < t)
            mask_30d = (times[prev] >= t - 30 * 86400) & (times[prev] < t)

            g.iloc[i, g.columns.get_loc("transactions_last_1h")] = np.sum(is_tx[prev] & mask_1h)
            g.iloc[i, g.columns.get_loc("transactions_last_24h")] = np.sum(is_tx[prev] & mask_24h)
            g.iloc[i, g.columns.get_loc("transactions_last_7d")] = np.sum(is_tx[prev] & mask_7d)
            g.iloc[i, g.columns.get_loc("transactions_last_30d")] = np.sum(is_tx[prev] & mask_30d)

            amt_24h = amounts[prev][is_tx[prev] & mask_24h]
            amt_7d = amounts[prev][is_tx[prev] & mask_7d]
            amt_30d = amounts[prev][is_tx[prev] & mask_30d]

            g.iloc[i, g.columns.get_loc("amount_sum_last_24h")] = np.sum(amt_24h)
            g.iloc[i, g.columns.get_loc("amount_sum_last_7d")] = np.sum(amt_7d)
            g.iloc[i, g.columns.get_loc("amount_sum_last_30d")] = np.sum(amt_30d)

            if len(amt_30d) > 0:
                g.iloc[i, g.columns.get_loc("amount_mean_last_30d")] = np.mean(amt_30d)
                g.iloc[i, g.columns.get_loc("amount_std_last_30d")] = np.std(amt_30d)

            g.iloc[i, g.columns.get_loc("logins_last_24h")] = np.sum(is_login[prev] & mask_24h)

        return g

    df = df.groupby("user_id", group_keys=False).apply(add_temporal_per_user)
    gc.collect()

    print("  → Transaction features...")
    df["transaction_description"] = df.apply(
        lambda r: f"{r['transaction_type'].upper()} payment" if r["event_type"] == "transaction" and pd.notna(
            r["transaction_type"]) else None,
        axis=1
    )
    df["beneficiary_bank_code"] = df.apply(
        lambda r: f"{np.random.randint(100, 999)}" if r["event_type"] == "transaction" else None,
        axis=1
    )
    mcc_pool = [5411, 5812, 5999, 6011, 7011, 4121]
    df["merchant_category_code"] = df.apply(
        lambda r: str(np.random.choice(mcc_pool)) if r["event_type"] == "transaction" and "card" in str(
            r.get("transaction_type", "")).lower() else None,
        axis=1
    )
    df["is_round_amount"] = df["amount"].fillna(0.0).apply(lambda x: int(x % 10 == 0 or x % 100 == 0))
    df["is_international"] = 0
    df["amount_increase_vs_30d_mean"] = (
            df["amount"].fillna(0.0) / df["amount_mean_last_30d"].replace(0, np.nan)).fillna(0.0)

    print("  → Location/network features...")
    df["ip_address"] = df.apply(lambda _: f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}", axis=1)
    df["ip_country"] = df["registered_country"]
    df["ip_region"] = df["registered_region"]
    df["ip_city"] = "São Paulo"
    df["ip_asn"] = "AS15169"
    df["ip_isp"] = np.random.choice(["Google LLC", "Amazon", "Cloudflare", "Local ISP"], size=len(df))
    df["is_vpn"] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
    df["is_datacenter_ip"] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
    df["geolocation_lat"] = -23.5505 + np.random.normal(0, 0.5, size=len(df))
    df["geolocation_lon"] = -46.6333 + np.random.normal(0, 0.5, size=len(df))
    df["ip_blacklisted"] = 0
    df["is_proxy"] = 0
    df["is_tor"] = 0
    df["connection_type"] = "broadband"
    df["is_location_anomaly"] = np.random.choice([0, 1], size=len(df), p=[0.97, 0.03])
    df["distance_from_registered_location_km"] = np.random.exponential(50, size=len(df))

    print("  → Device/environment features...")
    user_devices: Dict[str, List[DeviceProfile]] = {}

    def init_user_devices(user_id: str, channel_hint: str) -> List[DeviceProfile]:
        rng = random.Random(stable_hash(user_id))
        n = rng.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.13, 0.04], k=1)[0]
        return [make_device(rng, user_id=user_id, idx=i, channel_hint=channel_hint) for i in range(n)]

    device_data = []
    for i, row in df.iterrows():
        uid = row["user_id"]
        ch = row.get("channel", "web") or "web"

        if uid not in user_devices:
            user_devices[uid] = init_user_devices(uid, channel_hint=ch)

        rng = random.Random(stable_hash(f"{uid}:{int(pd.to_datetime(row['timestamp_utc']).timestamp())}"))
        pool = user_devices[uid]

        p_new = 0.012
        if row.get("user_risk_class", "") in {"high", "very_high"}:
            p_new *= 2.0
        if rng.random() < p_new and len(pool) < 8:
            pool.append(make_device(rng, user_id=uid, idx=len(pool), channel_hint=ch))

        chosen = rng.choice(pool)
        device_data.append({
            "device_id": chosen.device_id,
            "device_type": chosen.device_type,
            "os_family": chosen.os_family,
            "os_version": chosen.os_version,
            "browser_family": chosen.browser_family,
            "browser_version": chosen.browser_version,
            "app_version": chosen.app_version,
            "is_emulator": chosen.is_emulator,
            "is_rooted_or_jailbroken": chosen.is_rooted_or_jailbroken,
        })

    device_df = pd.DataFrame(device_data)
    for col in device_df.columns:
        df[col] = device_df[col].values

    def add_device_counts(g):
        g = g.copy()
        g["is_new_device_for_user"] = (~g["device_id"].astype(str).duplicated()).astype(int)
        g["registered_devices_count"] = g["device_id"].nunique()

        times = pd.to_datetime(g["timestamp_utc"]).values.astype("datetime64[s]").astype("int64")
        devs = g["device_id"].astype(str).values

        devices_30d = []
        for i in range(len(g)):
            t = times[i]
            prev = slice(0, i)
            mask = (times[prev] >= t - 30 * 86400) & (times[prev] < t)
            devices_30d.append(len(set(devs[prev][mask])))

        g["devices_last_30d"] = devices_30d
        g["active_devices_last_30d"] = devices_30d
        g["is_device_compromised"] = ((g["is_emulator"] == 1) | (g["is_rooted_or_jailbroken"] == 1)).astype(int)

        return g

    df = df.groupby("user_id", group_keys=False).apply(add_device_counts)
    gc.collect()

    print("  → Security/tech features...")
    df["tls_version"] = np.random.choice(["TLS1_2", "TLS1_3"], size=len(df), p=[0.25, 0.75])
    df["cipher_strength_bits"] = np.random.choice([128, 256], size=len(df), p=[0.15, 0.85])
    df["waf_present"] = np.random.choice([0, 1], size=len(df), p=[0.30, 0.70])
    df["waf_vendor"] = df["waf_present"].apply(lambda x: "AWS_WAF" if x == 1 else None)
    df["ids_present"] = np.random.choice([0, 1], size=len(df), p=[0.60, 0.40])
    df["ips_present"] = np.random.choice([0, 1], size=len(df), p=[0.50, 0.50])
    df["antimalware_present"] = np.random.choice([0, 1], size=len(df), p=[0.40, 0.60])
    df["antimalware_vendor"] = df["antimalware_present"].apply(lambda x: "GenericAV" if x == 1 else None)
    df["mfa_enabled"] = np.random.choice([0, 1], size=len(df), p=[0.35, 0.65])
    df["mfa_method"] = df["mfa_enabled"].apply(lambda x: np.random.choice(["sms", "totp", "push"]) if x == 1 else None)
    df["device_binding_enabled"] = np.random.choice([0, 1], size=len(df), p=[0.45, 0.55])
    df["risk_engine_version"] = "1.0.0"
    df["security_score_tech"] = np.random.uniform(0.5, 1.0, size=len(df))

    print("  → Fraud/history features...")
    df["sensitive_data_change_last_7d"] = 0
    df["previous_fraud_count"] = 0
    df["previous_chargeback_count"] = 0
    df["account_takeover_flag"] = 0
    df["velocity_alert_flag"] = 0
    df["blacklist_hit"] = np.random.choice([0, 1], size=len(df), p=[0.98, 0.02])
    df["whitelist_hit"] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    df["money_mule_score"] = 0.0
    df["device_fingerprint_match_count"] = 1
    df["ip_reputation_score"] = np.random.uniform(0.3, 0.9, size=len(df))

    print("  → Calculando fraud_probability...")
    user_risk_mult = df["user_risk_class"].map(FRAUD_MULTIPLIERS["by_user_risk_class"]).fillna(1.0)
    channel_mult = df["channel"].map(FRAUD_MULTIPLIERS["by_channel"]).fillna(1.0)
    tx_mult = df["transaction_type"].map(FRAUD_MULTIPLIERS["by_transaction_type"]).fillna(1.0)

    hour = df["hour_of_day"]
    hour_mult = pd.Series(1.0, index=df.index)
    hour_mult[hour < 6] = FRAUD_MULTIPLIERS["by_hour"]["night"]
    hour_mult[(hour >= 6) & (hour < 19)] = FRAUD_MULTIPLIERS["by_hour"]["business"]
    hour_mult[hour >= 19] = FRAUD_MULTIPLIERS["by_hour"]["evening"]

    dow_mult = pd.Series(FRAUD_MULTIPLIERS["by_dow"]["weekday"], index=df.index)
    dow_mult[df["day_of_week"] >= 5] = FRAUD_MULTIPLIERS["by_dow"]["weekend"]

    combined_mult = user_risk_mult * channel_mult * tx_mult * hour_mult * dow_mult
    mean_mult = combined_mult.mean()
    normalized_mult = combined_mult / mean_mult

    df["fraud_probability"] = (TARGET_FRAUD_RATE * normalized_mult).clip(upper=MAX_FRAUD_PROBABILITY)

    random_values = np.random.rand(len(df))
    df["is_fraud"] = (random_values < df["fraud_probability"]).astype(int)

    fraud_count = df["is_fraud"].sum()
    fraud_rate = fraud_count / len(df) * 100
    print(f"  → Fraudes: {fraud_count:,} ({fraud_rate:.2f}%)")

    scenario_names = list(FRAUD_SCENARIOS.keys())
    scenario_weights = list(FRAUD_SCENARIOS.values())
    total_weight = sum(scenario_weights)
    scenario_probs = [w / total_weight for w in scenario_weights]

    df["fraud_type"] = None
    fraud_mask = df["is_fraud"] == 1
    num_frauds = fraud_mask.sum()
    if num_frauds > 0:
        fraud_types = np.random.choice(scenario_names, size=num_frauds, p=scenario_probs)
        df.loc[fraud_mask, "fraud_type"] = fraud_types

    print("  → Text/LLM features...")
    df["message_text"] = df.apply(
        lambda r: f"Transaction of ${r['amount']:.2f}" if r["event_type"] == "transaction" and pd.notna(
            r["amount"]) else None,
        axis=1
    )
    df["message_length"] = df["message_text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df["message_language"] = "en"
    df["contains_url"] = 0
    df["contains_phone"] = 0
    df["num_special_chars"] = df["message_text"].apply(
        lambda x: sum(c in "!@#$%^&*()" for c in str(x)) if pd.notna(x) else 0)
    df["llm_risk_score"] = np.random.uniform(0.0, 1.0, size=len(df))
    df["llm_risk_reasoning"] = "No template"
    df["llm_phishing_detected"] = 0
    df["llm_sentiment_score"] = 0.0
    df["llm_urgency_score"] = 0.0

    print("✅ Features adicionadas")
    return df


# ============================================================================
# SCHEMA ENFORCER
# ============================================================================

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que todas as colunas necessárias existam e estejam na ordem correta."""
    print("\n[4/6] Aplicando schema final...")

    out = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[REQUIRED_COLUMNS]

    print(f"✅ Schema aplicado: {len(out.columns)} colunas")
    return out


# ============================================================================
# SALVAMENTO
# ============================================================================

def save_dataset(df: pd.DataFrame, output_dir: Path, chunk_size: int = 500_000):
    """Salva dataset em chunks."""
    print(f"\n[5/6] Salvando dataset em {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "dataset_full.csv"
    print(f"  → Salvando {len(df):,} linhas em {full_path.name}...")

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk.to_csv(full_path, mode=mode, header=header, index=False)
        if i % (chunk_size * 5) == 0:
            print(f"    Chunk {i // chunk_size + 1}/{(len(df) - 1) // chunk_size + 1}")

    print(f"✅ Dataset salvo: {full_path}")

    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD DETECTION DATASET - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Events: {len(df):,}\n")
        f.write(f"Total Columns: {len(df.columns)}\n")
        f.write(f"Date Range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}\n\n")

        if "is_fraud" in df.columns:
            fraud_count = df["is_fraud"].sum()
            f.write(f"Fraudulent: {fraud_count:,} ({fraud_count / len(df) * 100:.2f}%)\n")
            f.write(f"Legitimate: {len(df) - fraud_count:,} ({(len(df) - fraud_count) / len(df) * 100:.2f}%)\n\n")

        f.write("Event Types:\n")
        for et, count in df["event_type"].value_counts().items():
            f.write(f"  {et}: {count:,} ({count / len(df) * 100:.2f}%)\n")

    print(f"✅ Summary salvo: {summary_path}")


# ============================================================================
# GERAÇÃO DE FIGURAS ESTATÍSTICAS AVANÇADAS + TABELAS LATEX
# ============================================================================

def generate_advanced_figures(df: pd.DataFrame, output_dir: Path):
    """Gera figuras estatísticas avançadas + tabelas LaTeX para publicação."""
    print("\n[6/6] Gerando figuras estatísticas avançadas + tabelas LaTeX...")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = output_dir / "tables_latex"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["timestamp_utc"]).dt.date
    df_plot["week"] = pd.to_datetime(df_plot["timestamp_utc"]).dt.isocalendar().week
    df_plot["month"] = pd.to_datetime(df_plot["timestamp_utc"]).dt.month
    df_plot["year_month"] = pd.to_datetime(df_plot["timestamp_utc"]).dt.to_period("M")

    tx_df = df_plot[df_plot["event_type"] == "transaction"].copy()
    tx_df["log_amount"] = np.log1p(tx_df["amount"].fillna(0))

    # ========================================================================
    # FIGURA 1: Volume e Taxa de Fraude Diária com MA7
    # ========================================================================
    print("  → [1/20] Volume e taxa de fraude diária...")
    daily = df_plot.groupby("date").agg({
        "event_id": "count",
        "is_fraud": ["sum", "mean"]
    }).reset_index()
    daily.columns = ["date", "volume", "fraud_count", "fraud_rate"]
    daily["fraud_rate_pct"] = daily["fraud_rate"] * 100
    daily["volume_ma7"] = daily["volume"].rolling(7, min_periods=1).mean()
    daily["fraud_rate_ma7"] = daily["fraud_rate_pct"].rolling(7, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(daily["date"], daily["volume"], alpha=0.3, color="steelblue", label="Daily Volume")
    ax1.plot(daily["date"], daily["volume_ma7"], linewidth=2, color="darkblue", label="7-Day MA")
    ax1.set_ylabel("Event Volume", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(daily["date"], daily["fraud_rate_pct"], alpha=0.3, color="coral", label="Daily Fraud Rate")
    ax2.plot(daily["date"], daily["fraud_rate_ma7"], linewidth=2, color="darkred", label="7-Day MA")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "01_daily_volume_fraud_rate_ma7.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 2: Heatmap Fraude por Dia da Semana × Hora (MELHORADO)
    # ========================================================================
    print("  → [2/20] Heatmap fraude por dia/hora (com suporte mínimo)...")
    heat = tx_df.copy()
    grp = heat.groupby(["day_of_week", "hour_of_day"]).agg(
        fraud_rate=("is_fraud", "mean"),
        n=("is_fraud", "size"),
    ).reset_index()
    heatmap_rate = grp.pivot(index="day_of_week", columns="hour_of_day", values="fraud_rate") * 100
    heatmap_n = grp.pivot(index="day_of_week", columns="hour_of_day", values="n")

    min_n = max(50, int(0.0005 * len(heat)))
    mask = (heatmap_n.fillna(0) < min_n)

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heatmap_rate,
        cmap="YlOrRd",
        annot=False,
        fmt=".1f",
        cbar_kws={"label": "Fraud Rate (%)"},
        ax=ax,
        mask=mask,
        vmin=np.nanpercentile(heatmap_rate.values, 5),
        vmax=np.nanpercentile(heatmap_rate.values, 95),
    )
    ax.set_yticklabels(dow_labels, rotation=0)
    ax.set_xlabel("Hour of Day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Day of Week", fontsize=11, fontweight="bold")
    ax.set_title(f"Fraud Rate Heatmap (Transactions): DOW × Hour  |  masked if N<{min_n}", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_fraud_heatmap_dow_hour.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 3: ECDF de log(amount) (MELHORADO)
    # ========================================================================
    print("  → [3/20] ECDF de log(amount) (com KS e medianas)...")
    fraud_amounts = tx_df[tx_df["is_fraud"] == 1]["log_amount"].dropna()
    legit_amounts = tx_df[tx_df["is_fraud"] == 0]["log_amount"].dropna()

    max_points = 200_000
    if len(fraud_amounts) > max_points:
        fraud_amounts = fraud_amounts.sample(max_points, random_state=42)
    if len(legit_amounts) > max_points:
        legit_amounts = legit_amounts.sample(max_points, random_state=42)

    fraud_sorted = np.sort(fraud_amounts.values)
    legit_sorted = np.sort(legit_amounts.values)
    y_f = np.linspace(0, 1, len(fraud_sorted), endpoint=True)
    y_l = np.linspace(0, 1, len(legit_sorted), endpoint=True)

    ks_stat, ks_p = stats.ks_2samp(fraud_sorted, legit_sorted)
    med_f = float(np.median(fraud_sorted)) if len(fraud_sorted) else float('nan')
    med_l = float(np.median(legit_sorted)) if len(legit_sorted) else float('nan')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fraud_sorted, y_f, label=f"Fraud (n={len(fraud_sorted):,})", color="red", linewidth=2)
    ax.plot(legit_sorted, y_l, label=f"Legit (n={len(legit_sorted):,})", color="green", linewidth=2)
    ax.axvline(med_f, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.axvline(med_l, color="green", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.set_xlabel("log(1 + amount)", fontsize=11, fontweight="bold")
    ax.set_ylabel("ECDF", fontsize=11, fontweight="bold")
    ax.set_title(f"ECDF: log(1+amount) by Fraud  |  KS={ks_stat:.3f} (p={ks_p:.2e})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "03_ecdf_log_amount_by_fraud.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 4: Boxplot de log(amount) por mês
    # ========================================================================
    print("  → [4/20] Boxplot amount por mês...")
    tx_df["month"] = pd.to_datetime(tx_df["timestamp_utc"]).dt.month

    fig, ax = plt.subplots(figsize=(12, 6))
    tx_df.boxplot(column="log_amount", by="month", ax=ax, patch_artist=True)
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("log(1 + amount)", fontsize=11, fontweight="bold")
    ax.set_title("Transaction Amount Distribution by Month", fontsize=13, fontweight="bold")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(figures_dir / "04_amount_boxplot_by_month.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 5: Calibration Curve
    # ========================================================================
    print("  → [5/20] Calibration curve...")
    fraud_prob = df_plot["fraud_probability"].fillna(0).values
    is_fraud = df_plot["is_fraud"].values

    n_bins = 10
    bins = np.linspace(0, fraud_prob.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = []
    bin_means = []

    for i in range(n_bins):
        mask = (fraud_prob >= bins[i]) & (fraud_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_counts.append(mask.sum())
            bin_means.append(is_fraud[mask].mean())
        else:
            bin_counts.append(0)
            bin_means.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
    ax1.plot(bin_centers, bin_means, "o-", color="steelblue", linewidth=2, markersize=8, label="Model")
    ax1.set_xlabel("Predicted Fraud Probability", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Actual Fraud Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Calibration Curve", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(fraud_prob, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Fraud Probability", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_title("Distribution of Fraud Scores", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "05_calibration_curve.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 6: Top Features por AUC (numéricos)
    # ========================================================================
    print("  → [6/20] Top features por AUC...")
    numeric_cols = ["amount", "account_age_days", "transactions_last_24h", "amount_sum_last_24h",
                    "logins_last_24h", "devices_last_30d", "security_score_tech", "ip_reputation_score"]

    auc_scores = []
    for col in numeric_cols:
        if col in df_plot.columns:
            vals = df_plot[col].fillna(0).values
            if vals.std() > 0:
                try:
                    fpr, tpr, _ = roc_curve(is_fraud, vals)
                    auc_scores.append((col, auc(fpr, tpr)))
                except:
                    pass

    auc_scores = sorted(auc_scores, key=lambda x: abs(x[1] - 0.5), reverse=True)[:10]

    if auc_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        features, scores = zip(*auc_scores)
        colors = ["green" if s > 0.5 else "red" for s in scores]
        ax.barh(features, scores, color=colors, alpha=0.7, edgecolor="black")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Random (AUC=0.5)")
        ax.set_xlabel("AUC", fontsize=11, fontweight="bold")
        ax.set_title("Top Numeric Features by Univariate AUC", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(figures_dir / "06_top_features_by_auc.png", dpi=150)
        plt.close()

    # ========================================================================
    # FIGURA 7: Top Features por KS Statistic
    # ========================================================================
    print("  → [7/20] Top features por KS...")
    ks_scores = []
    for col in numeric_cols:
        if col in df_plot.columns:
            fraud_vals = df_plot[df_plot["is_fraud"] == 1][col].fillna(0).values
            legit_vals = df_plot[df_plot["is_fraud"] == 0][col].fillna(0).values
            if len(fraud_vals) > 0 and len(legit_vals) > 0:
                ks_stat, _ = stats.ks_2samp(fraud_vals, legit_vals)
                ks_scores.append((col, ks_stat))

    ks_scores = sorted(ks_scores, key=lambda x: x[1], reverse=True)[:10]

    if ks_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        features, scores = zip(*ks_scores)
        ax.barh(features, scores, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_xlabel("KS Statistic", fontsize=11, fontweight="bold")
        ax.set_title("Top Numeric Features by KS Statistic", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(figures_dir / "07_top_features_by_ks.png", dpi=150)
        plt.close()

    # ========================================================================
    # FIGURA 8: PSI Temporal (MELHORADO)
    # ========================================================================
    print("  → [8/20] PSI temporal (bins consistentes, ordenado)...")
    df_plot_sorted = df_plot.sort_values("timestamp_utc").reset_index(drop=True)
    mid = len(df_plot_sorted) // 2
    first_half = df_plot_sorted.iloc[:mid]
    second_half = df_plot_sorted.iloc[mid:]

    def calculate_psi(expected, actual, bins=10):
        expected_vals = expected.fillna(0).values
        actual_vals = actual.fillna(0).values

        if expected_vals.std() == 0 or actual_vals.std() == 0:
            return 0.0

        bin_edges = np.percentile(expected_vals, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        expected_counts, _ = np.histogram(expected_vals, bins=bin_edges)
        actual_counts, _ = np.histogram(actual_vals, bins=bin_edges)

        expected_pct = expected_counts / expected_counts.sum()
        actual_pct = actual_counts / actual_counts.sum()

        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    psi_scores = []
    for col in numeric_cols:
        if col in df_plot.columns:
            psi = calculate_psi(first_half[col], second_half[col])
            psi_scores.append((col, psi))

    psi_scores = sorted(psi_scores, key=lambda x: x[1], reverse=True)[:10]

    if psi_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        features, scores = zip(*psi_scores)
        colors = ["red" if s > 0.2 else "orange" if s > 0.1 else "green" for s in scores]
        ax.barh(features, scores, color=colors, alpha=0.7, edgecolor="black")
        ax.axvline(0.1, color="orange", linestyle="--", linewidth=1.5, label="PSI=0.1 (moderate)")
        ax.axvline(0.2, color="red", linestyle="--", linewidth=1.5, label="PSI=0.2 (high)")
        ax.set_xlabel("PSI", fontsize=11, fontweight="bold")
        ax.set_title("Population Stability Index: First Half vs Second Half", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(figures_dir / "08_psi_temporal.png", dpi=150)
        plt.close()

    # ========================================================================
    # FIGURA 9: Missingness por Feature Block (MELHORADO)
    # ========================================================================
    print("  → [9/20] Missingness por bloco...")
    feature_blocks = {
        "Event ID": ["event_id", "event_type", "event_source"],
        "User/Account": ["user_id", "account_id", "user_type", "user_segment"],
        "Temporal": ["timestamp_utc", "hour_of_day", "day_of_week"],
        "Transaction": ["amount", "currency", "transaction_type", "channel"],
        "Location": ["ip_address", "ip_country", "geolocation_lat", "geolocation_lon"],
        "Device": ["device_id", "device_type", "os_family", "browser_family"],
        "Security": ["tls_version", "mfa_enabled", "security_score_tech"],
        "Fraud": ["fraud_probability", "is_fraud", "fraud_type"],
    }

    missingness = []
    for block, cols in feature_blocks.items():
        existing_cols = [c for c in cols if c in df_plot.columns]
        if existing_cols:
            miss_pct = df_plot[existing_cols].isnull().mean().mean() * 100
            missingness.append((block, miss_pct))

    if missingness:
        fig, ax = plt.subplots(figsize=(10, 6))
        blocks, miss_pcts = zip(*missingness)
        ax.barh(blocks, miss_pcts, color="coral", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Missingness (%)", fontsize=11, fontweight="bold")
        ax.set_title("Missingness by Feature Block", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(figures_dir / "09_missingness_by_block.png", dpi=150)
        plt.close()

    # ========================================================================
    # FIGURA 10: Fraude por Segmento de Usuário
    # ========================================================================
    print("  → [10/20] Fraude por segmento...")
    segment_fraud = df_plot.groupby("user_segment")["is_fraud"].agg(["sum", "mean", "count"]).reset_index()
    segment_fraud["fraud_rate_pct"] = segment_fraud["mean"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(segment_fraud["user_segment"], segment_fraud["sum"], color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_ylabel("Fraud Count", fontsize=11, fontweight="bold")
    ax1.set_xlabel("User Segment", fontsize=11, fontweight="bold")
    ax1.set_title("Fraud Count by User Segment", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(segment_fraud["user_segment"], segment_fraud["fraud_rate_pct"], color="coral", alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("User Segment", fontsize=11, fontweight="bold")
    ax2.set_title("Fraud Rate by User Segment", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "10_fraud_by_segment.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 11: Fraude por Canal
    # ========================================================================
    print("  → [11/20] Fraude por canal...")
    channel_fraud = df_plot.groupby("channel")["is_fraud"].agg(["sum", "mean", "count"]).reset_index()
    channel_fraud["fraud_rate_pct"] = channel_fraud["mean"] * 100
    channel_fraud = channel_fraud.sort_values("fraud_rate_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(channel_fraud))
    width = 0.35

    ax.bar(x - width / 2, channel_fraud["sum"], width, label="Fraud Count", color="steelblue", alpha=0.7,
           edgecolor="black")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, channel_fraud["fraud_rate_pct"], width, label="Fraud Rate (%)", color="coral", alpha=0.7,
            edgecolor="black")

    ax.set_xlabel("Channel", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraud Count", fontsize=11, fontweight="bold", color="steelblue")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_fraud["channel"], rotation=45, ha="right")
    ax.set_title("Fraud Analysis by Channel", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "11_fraud_by_channel.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 12: Distribuição de Tipos de Fraude
    # ========================================================================
    print("  → [12/20] Distribuição de tipos de fraude...")
    fraud_types = df_plot[df_plot["is_fraud"] == 1]["fraud_type"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(fraud_types)))
    wedges, texts, autotexts = ax.pie(
        fraud_types.values,
        labels=fraud_types.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"}
    )
    ax.set_title("Distribution of Fraud Types", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "12_fraud_type_distribution.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 13: WOE/IV Analysis (Weight of Evidence / Information Value)
    # ========================================================================
    print("  → [13/20] WOE/IV analysis...")

    def calculate_woe_iv(df, feature, target, bins=10):
        """Calcula WOE e IV para uma feature numérica."""
        df_temp = df[[feature, target]].copy()
        df_temp = df_temp.dropna()

        if len(df_temp) == 0 or df_temp[feature].std() == 0:
            return None, None

        df_temp['bin'] = pd.qcut(df_temp[feature], q=bins, duplicates='drop')

        grouped = df_temp.groupby('bin', observed=True)[target].agg(['sum', 'count'])
        grouped['non_fraud'] = grouped['count'] - grouped['sum']
        grouped['fraud'] = grouped['sum']

        total_fraud = grouped['fraud'].sum()
        total_non_fraud = grouped['non_fraud'].sum()

        if total_fraud == 0 or total_non_fraud == 0:
            return None, None

        grouped['fraud_pct'] = grouped['fraud'] / total_fraud
        grouped['non_fraud_pct'] = grouped['non_fraud'] / total_non_fraud

        grouped['fraud_pct'] = grouped['fraud_pct'].replace(0, 0.0001)
        grouped['non_fraud_pct'] = grouped['non_fraud_pct'].replace(0, 0.0001)

        grouped['woe'] = np.log(grouped['fraud_pct'] / grouped['non_fraud_pct'])
        grouped['iv'] = (grouped['fraud_pct'] - grouped['non_fraud_pct']) * grouped['woe']

        iv_total = grouped['iv'].sum()

        return grouped, iv_total

    woe_features = ["amount", "account_age_days", "transactions_last_24h", "devices_last_30d"]
    iv_scores = []

    for feat in woe_features:
        if feat in tx_df.columns:
            _, iv = calculate_woe_iv(tx_df, feat, "is_fraud", bins=10)
            if iv is not None:
                iv_scores.append((feat, iv))

    if iv_scores:
        iv_scores = sorted(iv_scores, key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        features, ivs = zip(*iv_scores)
        colors = ["darkgreen" if iv > 0.3 else "green" if iv > 0.1 else "orange" if iv > 0.02 else "red" for iv in ivs]
        ax.barh(features, ivs, color=colors, alpha=0.7, edgecolor="black")
        ax.axvline(0.02, color="orange", linestyle="--", linewidth=1.5, label="IV=0.02 (weak)")
        ax.axvline(0.1, color="green", linestyle="--", linewidth=1.5, label="IV=0.1 (medium)")
        ax.axvline(0.3, color="darkgreen", linestyle="--", linewidth=1.5, label="IV=0.3 (strong)")
        ax.set_xlabel("Information Value (IV)", fontsize=11, fontweight="bold")
        ax.set_title("Information Value by Feature\nIV = Σ(fraud% - legit%) × ln(fraud%/legit%)", fontsize=13,
                     fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(figures_dir / "13_woe_iv_analysis.png", dpi=150)
        plt.close()

    # ========================================================================
    # FIGURA 14: Precision-Recall Curve
    # ========================================================================
    print("  → [14/20] Precision-Recall curve...")

    precision, recall, _ = precision_recall_curve(is_fraud, fraud_prob)
    ap_score = average_precision_score(is_fraud, fraud_prob)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, color="darkblue", linewidth=2, label=f"PR Curve (AP={ap_score:.3f})")
    ax.axhline(is_fraud.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Baseline (fraud rate={is_fraud.mean():.3f})")
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "14_precision_recall_curve.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 15: Lift Chart
    # ========================================================================
    print("  → [15/20] Lift chart...")

    df_lift = pd.DataFrame({
        'fraud_prob': fraud_prob,
        'is_fraud': is_fraud
    }).sort_values('fraud_prob', ascending=False).reset_index(drop=True)

    df_lift['decile'] = pd.qcut(df_lift.index, q=10, labels=False, duplicates='drop') + 1

    lift_data = df_lift.groupby('decile').agg({
        'is_fraud': ['sum', 'count', 'mean']
    }).reset_index()
    lift_data.columns = ['decile', 'fraud_count', 'total', 'fraud_rate']
    lift_data['lift'] = lift_data['fraud_rate'] / is_fraud.mean()
    lift_data['cumulative_fraud'] = lift_data['fraud_count'].cumsum()
    lift_data['cumulative_total'] = lift_data['total'].cumsum()
    lift_data['cumulative_fraud_rate'] = lift_data['cumulative_fraud'] / lift_data['cumulative_total']
    lift_data['cumulative_lift'] = lift_data['cumulative_fraud_rate'] / is_fraud.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(lift_data['decile'], lift_data['lift'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline (Lift=1)')
    ax1.set_xlabel('Decile (10=highest score)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Lift', fontsize=11, fontweight='bold')
    ax1.set_title('Lift by Decile', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    ax2.plot(lift_data['decile'], lift_data['cumulative_lift'], marker='o', color='darkblue', linewidth=2, markersize=8)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Decile (10=highest score)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Lift', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Lift', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "15_lift_chart.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 16: Gain Chart
    # ========================================================================
    print("  → [16/20] Gain chart...")

    fig, ax = plt.subplots(figsize=(10, 6))

    pct_population = lift_data['cumulative_total'] / lift_data['cumulative_total'].max() * 100
    pct_fraud_captured = lift_data['cumulative_fraud'] / lift_data['cumulative_fraud'].max() * 100

    ax.plot(pct_population, pct_fraud_captured, marker='o', color='darkgreen', linewidth=2, markersize=8, label='Model')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Random (Baseline)')
    ax.fill_between(pct_population, pct_fraud_captured, pct_population, alpha=0.2, color='green')
    ax.set_xlabel('% of Population (sorted by score)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% of Fraud Captured', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Gain Chart', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "16_gain_chart.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 17: ROC Curve
    # ========================================================================
    print("  → [17/20] ROC curve...")

    fpr, tpr, thresholds = roc_curve(is_fraud, fraud_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color='darkblue', linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC=0.5)')
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "17_roc_curve.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 18: Velocity Analysis (transactions_last_1h, transactions_last_24h)
    # ========================================================================
    print("  → [18/20] Velocity analysis...")

    velocity_features = ['transactions_last_1h', 'transactions_last_24h']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, feat in enumerate(velocity_features):
        if feat in df_plot.columns:
            fraud_vals = df_plot[df_plot['is_fraud'] == 1][feat].fillna(0)
            legit_vals = df_plot[df_plot['is_fraud'] == 0][feat].fillna(0)

            ks_stat, ks_p = stats.ks_2samp(fraud_vals, legit_vals)

            axes[idx].hist(legit_vals, bins=50, alpha=0.5, color='green', label=f'Legit (n={len(legit_vals):,})',
                           density=True)
            axes[idx].hist(fraud_vals, bins=50, alpha=0.5, color='red', label=f'Fraud (n={len(fraud_vals):,})',
                           density=True)
            axes[idx].axvline(legit_vals.median(), color='green', linestyle='--', linewidth=2, alpha=0.7)
            axes[idx].axvline(fraud_vals.median(), color='red', linestyle='--', linewidth=2, alpha=0.7)
            axes[idx].set_xlabel(feat, fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Density', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{feat}\nKS={ks_stat:.3f} (p={ks_p:.2e})', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "18_velocity_analysis.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 19: Feature Importance (Decision Tree Surrogate)
    # ========================================================================
    print("  → [19/20] Feature importance (decision tree surrogate)...")

    feature_cols = ["amount", "account_age_days", "transactions_last_24h", "amount_sum_last_24h",
                    "logins_last_24h", "devices_last_30d", "security_score_tech", "ip_reputation_score"]

    X = df_plot[feature_cols].fillna(0).values
    y = df_plot['is_fraud'].values

    # Amostra para performance
    if len(X) > 100_000:
        sample_idx = np.random.choice(len(X), 100_000, replace=False)
        X = X[sample_idx]
        y = y[sample_idx]

    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, random_state=42)
    tree.fit(X, y)

    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_cols[i] for i in indices], importances[indices], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Importance (Gini)', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance (Decision Tree Surrogate, max_depth=5)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(figures_dir / "19_feature_importance_tree.png", dpi=150)
    plt.close()

    # ========================================================================
    # FIGURA 20: Brier Score Decomposition
    # ========================================================================
    print("  → [20/20] Brier score decomposition...")

    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(is_fraud, fraud_prob)

    # Decomposição: Brier = Reliability - Resolution + Uncertainty
    # Reliability: calibration error
    # Resolution: ability to separate classes
    # Uncertainty: inherent uncertainty (fraud rate * (1 - fraud rate))

    uncertainty = is_fraud.mean() * (1 - is_fraud.mean())

    # Simplified resolution (variance of predicted probabilities weighted by actual outcomes)
    resolution = np.mean((fraud_prob - is_fraud.mean()) ** 2)

    # Reliability (calibration error)
    reliability = brier - uncertainty + resolution

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Brier score
    ax1.bar(['Brier Score'], [brier], color='steelblue', alpha=0.7, edgecolor='black', width=0.5)
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'Brier Score = {brier:.4f}\n(lower is better)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    # Decomposition
    components = ['Uncertainty\n(baseline)', 'Resolution\n(discrimination)', 'Reliability\n(calibration)']
    values = [uncertainty, resolution, reliability]
    colors = ['gray', 'green', 'red']

    ax2.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Component Value', fontsize=11, fontweight='bold')
    ax2.set_title('Brier Score Decomposition\nBrier = Uncertainty - Resolution + Reliability', fontsize=13,
                  fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figures_dir / "20_brier_score_decomposition.png", dpi=150)
    plt.close()

    # ========================================================================
    # TABELAS LaTeX
    # ========================================================================
    print("  → [T] Gerando tabelas LaTeX...")

    # T1: Sumário geral
    summary = pd.DataFrame({
        "metric": [
            "Total events",
            "Total transactions",
            "Fraud events",
            "Fraud rate (events)",
            "Fraud transactions",
            "Fraud rate (transactions)",
            "Start timestamp",
            "End timestamp",
        ],
        "value": [
            len(df_plot),
            int((df_plot["event_type"] == "transaction").sum()),
            int(df_plot["is_fraud"].sum()),
            float(df_plot["is_fraud"].mean()),
            int(tx_df["is_fraud"].sum()),
            float(tx_df["is_fraud"].mean()) if len(tx_df) else float('nan'),
            str(pd.to_datetime(df_plot["timestamp_utc"]).min()),
            str(pd.to_datetime(df_plot["timestamp_utc"]).max()),
        ]
    })
    (tables_dir / "table_01_summary.tex").write_text(
        to_latex_table(summary, caption="Dataset summary.", label="tab:dataset_summary", index=False),
        encoding="utf-8",
    )

    # T2: Schema (coluna, tipo, missing)
    schema = pd.DataFrame({
        "column": df_plot.columns,
        "dtype": [str(t) for t in df_plot.dtypes],
        "missing_rate": df_plot.isna().mean().values,
    }).sort_values("missing_rate", ascending=False)
    (tables_dir / "table_02_schema_missingness.tex").write_text(
        to_latex_table(schema.head(40), caption="Top 40 columns by missingness.", label="tab:schema_missingness",
                       index=False),
        encoding="utf-8",
    )

    # T3: Fraude por segmento
    seg = df_plot.groupby("user_segment").agg(
        n=("event_id", "size"),
        fraud_n=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
    ).reset_index().sort_values("fraud_rate", ascending=False)
    (tables_dir / "table_03_fraud_by_segment.tex").write_text(
        to_latex_table(seg, caption="Fraud by user segment.", label="tab:fraud_by_segment", index=False),
        encoding="utf-8",
    )

    # T4: Fraude por canal (apenas transações)
    ch = tx_df.groupby("channel").agg(
        n=("event_id", "size"),
        fraud_n=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
        median_amount=("amount", "median"),
    ).reset_index().sort_values("fraud_rate", ascending=False)
    (tables_dir / "table_04_fraud_by_channel_tx.tex").write_text(
        to_latex_table(ch, caption="Fraud by channel (transactions only).", label="tab:fraud_by_channel_tx",
                       index=False),
        encoding="utf-8",
    )

    # T5: Top univariate AUC
    if auc_scores:
        auc_df = pd.DataFrame(auc_scores, columns=["feature", "auc"]).sort_values("auc", ascending=False)
        (tables_dir / "table_05_top_univariate_auc.tex").write_text(
            to_latex_table(auc_df, caption="Top numeric features by univariate AUC.", label="tab:top_univariate_auc",
                           index=False),
            encoding="utf-8",
        )

    # T6: Top KS
    if ks_scores:
        ks_df = pd.DataFrame(ks_scores, columns=["feature", "ks_statistic"]).sort_values("ks_statistic",
                                                                                         ascending=False)
        (tables_dir / "table_06_top_ks.tex").write_text(
            to_latex_table(ks_df, caption="Top numeric features by KS statistic.", label="tab:top_ks", index=False),
            encoding="utf-8",
        )

    # T7: PSI
    if psi_scores:
        psi_df = pd.DataFrame(psi_scores, columns=["feature", "psi"]).sort_values("psi", ascending=False)
        (tables_dir / "table_07_psi_temporal.tex").write_text(
            to_latex_table(psi_df, caption="Population Stability Index (first half vs second half).",
                           label="tab:psi_temporal", index=False),
            encoding="utf-8",
        )

    # T8: IV scores
    if iv_scores:
        iv_df = pd.DataFrame(iv_scores, columns=["feature", "information_value"]).sort_values("information_value",
                                                                                              ascending=False)
        (tables_dir / "table_08_information_value.tex").write_text(
            to_latex_table(iv_df, caption="Information Value by feature.", label="tab:information_value", index=False),
            encoding="utf-8",
        )

    # T9: Lift by decile
    (tables_dir / "table_09_lift_by_decile.tex").write_text(
        to_latex_table(lift_data[['decile', 'fraud_count', 'total', 'fraud_rate', 'lift', 'cumulative_lift']],
                       caption="Lift analysis by decile.", label="tab:lift_by_decile", index=False),
        encoding="utf-8",
    )

    # T10: Model performance summary
    perf_summary = pd.DataFrame({
        "metric": ["ROC AUC", "Average Precision", "Brier Score", "Fraud Rate (baseline)"],
        "value": [roc_auc, ap_score, brier, is_fraud.mean()]
    })
    (tables_dir / "table_10_model_performance.tex").write_text(
        to_latex_table(perf_summary, caption="Model performance summary.", label="tab:model_performance", index=False),
        encoding="utf-8",
    )

    print(f"✅ 20 figuras + 10 tabelas LaTeX salvas")
    print(f"   Figuras: {figures_dir}")
    print(f"   Tabelas: {tables_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gera dataset sintético completo de detecção de fraude")
    parser.add_argument("--output-dir", type=str, default="output_v3", help="Diretório de saída")
    parser.add_argument("--num-users", type=int, default=50000, help="Número de usuários")
    parser.add_argument("--year", type=int, default=2025, help="Ano para geração de dados")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = make_versioned_output_dir(Path(args.output_dir))
    start_date = datetime(args.year, 1, 1)
    end_date = datetime(args.year + 1, 1, 1)

    print("=" * 80)
    print("GERAÇÃO DE DATASET SINTÉTICO DE DETECÇÃO DE FRAUDE")
    print("=" * 80)
    print(f"Usuários: {args.num_users:,}")
    print(f"Período: {start_date.date()} até {end_date.date()}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    users_df = generate_users(args.num_users, start_date, seed=args.seed)
    events_df = generate_events(users_df, start_date, end_date, seed=args.seed)
    df = add_all_features(events_df, users_df, seed=args.seed)
    del events_df
    gc.collect()

    df = enforce_schema(df)
    save_dataset(df, output_dir)
    generate_advanced_figures(df, output_dir)

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("=" * 80)
    print(f"Dataset: {output_dir / 'dataset_full.csv'}")
    print(f"Figuras: {output_dir / 'figures'} (20 figuras)")
    print(f"Tabelas: {output_dir / 'tables_latex'} (10 tabelas LaTeX)")
    print("=" * 80)


if __name__ == "__main__":
    main()