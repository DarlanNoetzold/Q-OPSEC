from __future__ import annotations

import random
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from faker import Faker

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger


logger = get_logger("recipient_generator")


class RecipientGenerator:
    """Generate synthetic recipient pool and assign recipients to transactions.

    This supports section 1.4 (Transaction / Business Content):
      - recipient_id
      - recipient_type (individual, business, government, etc.)
      - recipient_account_age_days
      - recipient_country
      - is_new_recipient (computed per user)
      - transactions_with_recipient_last_30d (computed per user)

    Strategy:
      1. Generate a global recipient pool.
      2. For each transaction event, assign a recipient based on user profile & fraud scenario.
      3. Track user->recipient history for "is_new_recipient" logic.
    """

    def __init__(self, random_seed: int | None = None) -> None:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.faker = Faker()
        self.faker.seed_instance(random_seed or 0)

        self.dataset_cfg = default_config_loader.load("dataset_config.yaml")
        self.user_profiles_cfg = default_config_loader.load("user_profiles.yaml")

        # Build recipient pool
        self.recipients: List[Dict] = self._generate_recipient_pool()
        logger.info("Generated {n} recipients", n=len(self.recipients))

    def _generate_recipient_pool(self, pool_size: int = 5000) -> List[Dict]:
        """Create a pool of synthetic recipients."""
        recipients = []

        recipient_types = [
            ("individual", 0.70),
            ("business", 0.20),
            ("government", 0.05),
            ("ngo", 0.03),
            ("unknown", 0.02),
        ]
        types, weights = zip(*recipient_types)

        countries = ["BR", "US", "AR", "MX", "CO", "CL"]

        for i in range(pool_size):
            recipient_id = f"R{i:07d}"
            recipient_type = random.choices(types, weights=weights, k=1)[0]

            # recipient_account_age_days: synthetic, 0 to 3650 days
            account_age = random.randint(0, 3650)

            recipient_country = random.choices(
                countries,
                weights=[0.60, 0.15, 0.10, 0.05, 0.05, 0.05],
                k=1
            )[0]

            recipients.append({
                "recipient_id": recipient_id,
                "recipient_type": recipient_type,
                "recipient_account_age_days": account_age,
                "recipient_country": recipient_country,
            })

        return recipients

    def assign_recipients_to_transactions(
        self,
        events: pd.DataFrame,
        users: pd.DataFrame,
    ) -> pd.DataFrame:
        """For each transaction event, assign a recipient.

        Also compute:
          - is_new_recipient (per user)
          - transactions_with_recipient_last_30d (per user)

        Returns updated events DataFrame.
        """
        df = events.copy()

        # Initialize columns for ALL events (not just transactions)
        df["recipient_id"] = None
        df["recipient_type"] = None
        df["recipient_account_age_days"] = np.nan
        df["recipient_country"] = None
        df["is_new_recipient"] = 0
        df["transactions_with_recipient_last_30d"] = 0

        # Filter only transaction events
        tx_mask = df["event_type"] == "transaction"
        if not tx_mask.any():
            logger.warning("No transaction events to assign recipients")
            return df

        # Sort by user and time for sequential processing
        df = df.sort_values(["user_id", "timestamp_utc"]).reset_index(drop=True)

        # Track user recipient history: user_id -> Set[recipient_id]
        user_recipient_history: Dict[str, Set[str]] = {}

        # Track recent transactions with each recipient for rolling window
        # user_id -> recipient_id -> List[timestamp_sec]
        user_recipient_tx_times: Dict[str, Dict[str, List[int]]] = {}

        # Convert timestamps to seconds for easier computation
        timestamps_sec = df["timestamp_utc"].astype("int64") // 10**9

        for idx, row in df.iterrows():
            if row["event_type"] != "transaction":
                continue

            user_id = row["user_id"]
            t_sec = timestamps_sec[idx]

            # Select a recipient
            # For normal users: mostly recurring recipients with some new ones
            # For fraudsters: higher rate of new recipients (from config)
            profile_name = users.loc[users["user_id"] == user_id, "profile_name"].iloc[0]
            profile_cfg = self.user_profiles_cfg["profiles"][profile_name]
            new_recipient_rate = profile_cfg["transaction_behavior"].get("new_recipient_rate", 0.1)

            # Initialize history if needed
            if user_id not in user_recipient_history:
                user_recipient_history[user_id] = set()
            if user_id not in user_recipient_tx_times:
                user_recipient_tx_times[user_id] = {}

            known_recipients = list(user_recipient_history[user_id])

            # Decide if new or recurring
            if not known_recipients or random.random() < new_recipient_rate:
                # Pick a new recipient from pool
                recipient = random.choice(self.recipients)
                is_new = 1
            else:
                # Pick a known recipient (biased towards recent ones)
                recipient_id_choice = random.choice(known_recipients)
                recipient = next(r for r in self.recipients if r["recipient_id"] == recipient_id_choice)
                is_new = 0

            recipient_id = recipient["recipient_id"]

            # Update history
            user_recipient_history[user_id].add(recipient_id)

            # Track this transaction timestamp for recipient
            if recipient_id not in user_recipient_tx_times[user_id]:
                user_recipient_tx_times[user_id][recipient_id] = []
            user_recipient_tx_times[user_id][recipient_id].append(t_sec)

            # Compute transactions_with_recipient_last_30d
            win_30d = t_sec - 30 * 86400
            recent_tx = [
                t for t in user_recipient_tx_times[user_id][recipient_id]
                if t >= win_30d and t < t_sec
            ]
            tx_last_30d = len(recent_tx)

            # Assign to DataFrame
            df.at[idx, "recipient_id"] = recipient_id
            df.at[idx, "recipient_type"] = recipient["recipient_type"]
            df.at[idx, "recipient_account_age_days"] = recipient["recipient_account_age_days"]
            df.at[idx, "recipient_country"] = recipient["recipient_country"]
            df.at[idx, "is_new_recipient"] = is_new
            df.at[idx, "transactions_with_recipient_last_30d"] = tx_last_30d

        logger.info("Assigned recipients to {n} transactions", n=tx_mask.sum())
        return df