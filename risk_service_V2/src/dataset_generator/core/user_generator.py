from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dateutil import tz
from faker import Faker

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger


logger = get_logger("user_generator")


@dataclass
class UserProfileConfig:
    name: str
    weight: float
    user_type: str
    user_segment: str
    base_user_risk_class: str
    country_pool: List[str]
    timezone_pool: List[str]
    events_per_month_mean: float
    events_per_month_std: float
    transaction_behavior: Dict
    device_profile: Dict


class UserGenerator:
    """Generate synthetic users aligned with section 1.2 (User / Account).

    Fields produced per user:
      - user_id
      - account_id
      - user_type
      - user_segment
      - user_risk_class
      - account_creation_date
      - account_age_days (computed later given event timestamp)
      - registered_country
      - registered_region (synthetic)
      - timezone
      - base profile info used later for behavior
    """

    def __init__(self, random_seed: int | None = None) -> None:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.config = default_config_loader.load("user_profiles.yaml")
        self.dataset_cfg = default_config_loader.load("dataset_config.yaml")

        self.faker = Faker()
        self.faker.seed_instance(random_seed or 0)

        self.profiles: List[UserProfileConfig] = self._load_profiles()
        self.profile_weights = [p.weight for p in self.profiles]

        logger.info(
            "Loaded {n} user profiles: {names}",
            n=len(self.profiles),
            names=[p.name for p in self.profiles],
        )

    def _load_profiles(self) -> List[UserProfileConfig]:
        profiles_cfg = self.config.get("profiles", {})
        profiles: List[UserProfileConfig] = []
        for name, cfg in profiles_cfg.items():
            profiles.append(
                UserProfileConfig(
                    name=name,
                    weight=float(cfg.get("weight", 1.0)),
                    user_type=cfg["user_type"],
                    user_segment=cfg["user_segment"],
                    base_user_risk_class=cfg.get("base_user_risk_class", "medium"),
                    country_pool=cfg.get("country_pool", ["BR"]),
                    timezone_pool=cfg.get("timezone_pool", ["America/Sao_Paulo"]),
                    events_per_month_mean=float(cfg.get("events_per_month_mean", 30)),
                    events_per_month_std=float(cfg.get("events_per_month_std", 10)),
                    transaction_behavior=cfg.get("transaction_behavior", {}),
                    device_profile=cfg.get("device_profile", {}),
                )
            )
        return profiles

    def _sample_profile(self) -> UserProfileConfig:
        return random.choices(self.profiles, weights=self.profile_weights, k=1)[0]

    def _sample_account_creation_date(self, start_date: datetime, end_date: datetime) -> datetime:
        """Sample an account creation date before end_date, up to 3 years back."""
        max_age_days = 365 * 3
        end_minus = end_date - timedelta(days=random.randint(0, max_age_days))
        # Ensure not before global start_date
        if end_minus < start_date:
            end_minus = start_date
        return end_minus

    def generate_users(self) -> pd.DataFrame:
        """Generate users according to dataset_config.generation.num_users.

        Returns a DataFrame with one row per user.
        """
        num_users = int(
            self.dataset_cfg.get("generation", {}).get("num_users", 1000)
        )
        start_date_str = self.dataset_cfg["generation"]["start_date"]
        end_date_str = self.dataset_cfg["generation"]["end_date"]
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        records: List[Dict] = []
        for user_idx in range(num_users):
            profile = self._sample_profile()

            user_id = f"U{user_idx:07d}"
            account_id = f"A{user_idx:07d}"

            registered_country = random.choice(profile.country_pool)
            # synthetic region name using Faker
            registered_region = self.faker.state_abbr() if registered_country == "US" else self.faker.state()

            timezone_name = random.choice(profile.timezone_pool)

            account_creation_date = self._sample_account_creation_date(
                start_date=start_date,
                end_date=end_date,
            )

            record = {
                "user_id": user_id,
                "account_id": account_id,
                "user_type": profile.user_type,
                "user_segment": profile.user_segment,
                "user_risk_class": profile.base_user_risk_class,
                "registered_country": registered_country,
                "registered_region": registered_region,
                "timezone": timezone_name,
                "account_creation_date": account_creation_date.date(),
                # we'll compute account_age_days later per event based on event timestamp
                "profile_name": profile.name,
                "events_per_month_mean": profile.events_per_month_mean,
                "events_per_month_std": profile.events_per_month_std,
            }
            records.append(record)

        users_df = pd.DataFrame.from_records(records)
        logger.info("Generated {n} users", n=len(users_df))
        return users_df
