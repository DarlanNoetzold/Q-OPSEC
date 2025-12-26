from __future__ import annotations

import random

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger


logger = get_logger("security_tech_features")


def add_security_tech_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.7 Authentication / Authorization / Security technologies & versions.

    Uses config/security_tech.yaml to simulate:
      - tls_version, cipher_strength
      - waf_present, waf_vendor
      - ids_present, ips_present
      - antimalware_present, antimalware_vendor
      - mfa_enabled, mfa_method
      - device_binding_enabled
      - risk_engine_version
      - security_score_tech (synthetic score based on configuration)
    """

    df = events.copy()
    sec_cfg = default_config_loader.load("security_tech.yaml")

    # TLS
    tls_dist = sec_cfg.get("tls_versions", {"TLS1_3": 0.7, "TLS1_2": 0.3})
    tls_versions = list(tls_dist.keys())
    tls_weights = list(tls_dist.values())

    # WAF
    waf_cfg = sec_cfg.get("waf", {})
    waf_enabled_prob = float(waf_cfg.get("enabled_probability", 0.8))
    waf_vendors = waf_cfg.get("vendors", ["GenericWAF"])

    # IDS/IPS
    ids_cfg = sec_cfg.get("ids_ips", {})
    ids_enabled_prob = float(ids_cfg.get("ids_enabled_probability", 0.7))
    ips_enabled_prob = float(ids_cfg.get("ips_enabled_probability", 0.6))

    # Antimalware
    av_cfg = sec_cfg.get("antimalware", {})
    av_enabled_prob = float(av_cfg.get("enabled_probability", 0.9))
    av_vendors = av_cfg.get("vendors", ["GenericAV"])

    # MFA
    mfa_cfg = sec_cfg.get("mfa", {})
    mfa_enabled_prob = float(mfa_cfg.get("enabled_probability", 0.75))
    mfa_methods = mfa_cfg.get("methods", ["sms", "totp", "push"])

    # Device binding
    device_binding_prob = float(sec_cfg.get("device_binding_enabled_probability", 0.6))

    # Risk engine
    risk_engine_cfg = sec_cfg.get("risk_engine", {})
    risk_engine_versions = risk_engine_cfg.get("versions", ["1.0.0"])

    # Prepare columns with defaults
    defaults = {
        "tls_version": None,
        "cipher_strength_bits": np.nan,
        "waf_present": 0,
        "waf_vendor": None,
        "ids_present": 0,
        "ips_present": 0,
        "antimalware_present": 0,
        "antimalware_vendor": None,
        "mfa_enabled": 0,
        "mfa_method": None,
        "device_binding_enabled": 0,
        "risk_engine_version": None,
        "security_score_tech": np.nan,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Global assignments (could be refined by channel, event_type, etc.)
    n = len(df)

    # TLS version & cipher strength (simplified mapping)
    df["tls_version"] = random.choices(tls_versions, weights=tls_weights, k=n)

    strength_map = {"TLS1_3": 256, "TLS1_2": 128, "TLS1_1": 112, "TLS1_0": 80}
    df["cipher_strength_bits"] = df["tls_version"].map(strength_map).fillna(80)

    # WAF
    waf_present_flags = np.random.rand(n) < waf_enabled_prob
    df["waf_present"] = waf_present_flags.astype(int)
    df.loc[waf_present_flags, "waf_vendor"] = np.random.choice(waf_vendors, size=waf_present_flags.sum())

    # IDS/IPS
    ids_flags = np.random.rand(n) < ids_enabled_prob
    ips_flags = np.random.rand(n) < ips_enabled_prob
    df["ids_present"] = ids_flags.astype(int)
    df["ips_present"] = ips_flags.astype(int)

    # Antimalware
    av_flags = np.random.rand(n) < av_enabled_prob
    df["antimalware_present"] = av_flags.astype(int)
    df.loc[av_flags, "antimalware_vendor"] = np.random.choice(av_vendors, size=av_flags.sum())

    # MFA - mais relevante para eventos de login/transação
    mfa_flags = np.random.rand(n) < mfa_enabled_prob
    df["mfa_enabled"] = mfa_flags.astype(int)
    df.loc[mfa_flags, "mfa_method"] = np.random.choice(mfa_methods, size=mfa_flags.sum())

    # Device binding
    db_flags = np.random.rand(n) < device_binding_prob
    df["device_binding_enabled"] = db_flags.astype(int)

    # Risk engine version
    df["risk_engine_version"] = np.random.choice(risk_engine_versions, size=n)

    # security_score_tech: score sintético (0-1) baseado em stack de segurança
    # Quanto mais proteções presentes, maior o score
    protections_count = (
        df["waf_present"].astype(int)
        + df["ids_present"].astype(int)
        + df["ips_present"].astype(int)
        + df["antimalware_present"].astype(int)
        + df["mfa_enabled"].astype(int)
        + df["device_binding_enabled"].astype(int)
    )
    max_protections = 6.0
    df["security_score_tech"] = (protections_count / max_protections).clip(0, 1)

    logger.info("Security technology features (1.7) added")
    return df