from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("event_generator")


@dataclass
class EventMixConfig:
    transaction: float
    login: float
    password_change: float
    email_change: float
    phone_change: float
    address_change: float
    mfa_setup: float
    device_registration: float


class EventGenerator:
    """Generate raw events for users aligned with 1.1, and basic 1.3/1.4 structure.

    At this stage we generate:
      - event_id
      - user_id, account_id
      - event_type
      - event_source (approximate)
      - timestamp_utc
      - timezone (copied from user)
      - basic transaction fields: amount, currency, channel, transaction_type

    Detailed features will be filled by feature modules later.
    """

    def __init__(self, users_df: pd.DataFrame) -> None:
        """
        Initialize EventGenerator with users DataFrame.

        Args:
            users_df: DataFrame containing user information
        """
        self.users_df = users_df

        # Load configs
        self.dataset_cfg = default_config_loader.load("dataset_config.yaml")
        self.user_profiles_cfg = default_config_loader.load("user_profiles.yaml")

        # Set random seed if configured
        random_seed = self.dataset_cfg.get("random_seed", None)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.event_mix = self._load_event_mix()

    def _load_event_mix(self) -> Dict[str, float]:
        mix_cfg = self.dataset_cfg.get("generation", {}).get("event_mix", {})
        # Normalize just in case
        total = sum(mix_cfg.values()) or 1.0
        return {k: v / total for k, v in mix_cfg.items()}

    def _sample_num_events_for_user(self, profile_name: str, months: float) -> int:
        profile_cfg = self.user_profiles_cfg["profiles"][profile_name]
        mean = float(profile_cfg.get("events_per_month_mean", 30)) * months
        std = float(profile_cfg.get("events_per_month_std", 10)) * (months ** 0.5)
        n = max(1, int(np.random.normal(mean, std)))
        return n

    def _sample_event_type(self) -> str:
        event_types = list(self.event_mix.keys())
        weights = list(self.event_mix.values())
        return random.choices(event_types, weights=weights, k=1)[0]

    def _sample_timestamp_for_user(self, start_date: datetime, end_date: datetime) -> datetime:
        """
        Gera um timestamp com padrão temporal realista (não-uniforme).
        """
        temporal = self.dataset_cfg.get("temporal_patterns", {})
        hour_weights = temporal.get("events_by_hour_weights", None)

        # Se não houver padrão temporal configurado, usar distribuição uniforme
        if hour_weights is None:
            total_seconds = int((end_date - start_date).total_seconds())
            offset = random.randint(0, total_seconds)
            return start_date + timedelta(seconds=offset)

        # Normalizar pesos
        hour_weights = np.array(hour_weights, dtype=float)
        hour_weights = hour_weights / hour_weights.sum()

        # Gerar data uniforme
        total_days = (end_date - start_date).days
        day_offset = random.randint(0, total_days)
        base_date = start_date + timedelta(days=day_offset)

        # Gerar hora com distribuição ponderada
        hour = np.random.choice(np.arange(24), p=hour_weights)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        # Combinar
        timestamp = base_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            hours=int(hour), minutes=minute, seconds=second
        )

        return timestamp

    def _sample_transaction_details(self, user_row: pd.Series) -> Dict:
        profile_cfg = self.user_profiles_cfg["profiles"][user_row["profile_name"]]
        tx_cfg = profile_cfg["transaction_behavior"]

        amount = max(
            1.0,
            np.random.normal(tx_cfg.get("amount_mean", 200.0), tx_cfg.get("amount_std", 100.0)),
        )
        currency = tx_cfg.get("currency", "BRL")

        tx_type_dist = tx_cfg.get("transaction_type_distribution", {"pix": 1.0})
        tx_types = list(tx_type_dist.keys())
        tx_weights = list(tx_type_dist.values())
        transaction_type = random.choices(tx_types, weights=tx_weights, k=1)[0]

        channel_dist = tx_cfg.get("channel_distribution", {"mobile_android": 1.0})
        channels = list(channel_dist.keys())
        ch_weights = list(channel_dist.values())
        channel = random.choices(channels, weights=ch_weights, k=1)[0]

        return {
            "amount": float(amount),
            "currency": currency,
            "transaction_type": transaction_type,
            "channel": channel,
        }

    def generate_events(self) -> pd.DataFrame:
        """Generate events for each user.

        Returns a DataFrame of raw events with basic columns:
          - event_id, user_id, account_id, event_type, event_source,
            timestamp_utc, timezone,
            amount, currency, transaction_type, channel (for transactions only).
        """
        gen_cfg = self.dataset_cfg["generation"]
        start_date = datetime.fromisoformat(gen_cfg["start_date"])
        end_date = datetime.fromisoformat(gen_cfg["end_date"])

        total_months = max(1.0, (end_date - start_date).days / 30.0)

        records: List[Dict] = []
        event_counter = 0

        for _, user in self.users_df.iterrows():
            n_events = self._sample_num_events_for_user(user["profile_name"], total_months)

            for _ in range(n_events):
                event_type = self._sample_event_type()
                ts_utc = self._sample_timestamp_for_user(start_date, end_date)

                event_source = self._infer_event_source(event_type, user)

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
                    base_record.update(self._sample_transaction_details(user))
                else:
                    # For non-transaction, set NaNs for transaction fields
                    base_record.update(
                        {
                            "amount": np.nan,
                            "currency": None,
                            "transaction_type": None,
                            "channel": self._infer_channel_from_source(event_source),
                        }
                    )

                records.append(base_record)
                event_counter += 1

        events_df = pd.DataFrame.from_records(records)
        logger.info("Generated {} events", len(events_df))
        return events_df

    def _infer_event_source(self, event_type: str, user_row: pd.Series) -> str:
        # Simple heuristic for now
        if event_type in {"login", "transaction"}:
            profile_cfg = self.user_profiles_cfg["profiles"][user_row["profile_name"]]
            channel_dist = profile_cfg["transaction_behavior"].get(
                "channel_distribution", {"mobile_android": 1.0}
            )
            channels = list(channel_dist.keys())
            weights = list(channel_dist.values())
            channel = random.choices(channels, weights=weights, k=1)[0]

            if channel in {"mobile_android", "mobile_ios"}:
                return "mobile_app"
            elif channel == "web":
                return "web_app"
            elif channel == "atm":
                return "atm"
            elif channel == "call_center":
                return "call_center"
            else:
                return "api_partner"
        elif event_type in {"password_change", "email_change", "phone_change", "address_change", "mfa_setup"}:
            return "web_app"
        elif event_type == "device_registration":
            return "mobile_app"
        else:
            return "web_app"

    def _infer_channel_from_source(self, event_source: str) -> str:
        if event_source == "mobile_app":
            # randomly android/ios
            return random.choice(["mobile_android", "mobile_ios"])
        if event_source == "web_app":
            return "web"
        if event_source == "atm":
            return "atm"
        if event_source == "call_center":
            return "call_center"
        if event_source == "api_partner":
            return "api_partner"
        return "web"