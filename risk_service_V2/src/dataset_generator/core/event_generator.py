import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from faker import Faker
from src.common.logger import log


class EventGenerator:
    def __init__(self, users_df: pd.DataFrame, config: Dict[str, Any], random_seed: int = 42):
        """Initialize event generator.

        Args:
            users_df: DataFrame with user data
            config: Dataset configuration
            random_seed: Random seed
        """
        self.users_df = users_df
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.fake = Faker()
        Faker.seed(random_seed)

        self.start_date = datetime.strptime(config['generation']['start_date'], '%Y-%m-%d')
        self.time_period_days = config['generation']['time_period_days']
        self.end_date = self.start_date + timedelta(days=self.time_period_days)

    def generate_events(self) -> pd.DataFrame:
        """Generate all events for all users.

        Returns:
            DataFrame with events
        """
        log.info(f"Generating events for {len(self.users_df)} users...")
        log.info(f"  Time period: {self.start_date.date()} to {self.end_date.date()}")

        all_events = []

        for idx, user in self.users_df.iterrows():
            user_events = self._generate_user_events(user)
            all_events.extend(user_events)

            if (idx + 1) % 1000 == 0:
                log.info(f"  Processed {idx + 1}/{len(self.users_df)} users...")

        df = pd.DataFrame(all_events)
        df = df.sort_values('timestamp_utc').reset_index(drop=True)

        # Add event IDs
        df['event_id'] = [f"evt_{i:012d}" for i in range(len(df))]

        log.info(f"✓ Generated {len(df)} events")
        log.info(f"  Event types:\n{df['event_type'].value_counts()}")

        return df

    def _generate_user_events(self, user: pd.Series) -> List[Dict[str, Any]]:
        """Generate events for a single user.

        Args:
            user: User data series

        Returns:
            List of event dictionaries
        """
        events = []

        # Calculate number of events
        days_active = (self.end_date - max(user['account_creation_date'], self.start_date)).days
        if days_active <= 0:
            return events

        num_events = max(1, int(self.rng.normal(
            user['transactions_per_day_mean'] * days_active,
            user['transactions_per_day_std'] * np.sqrt(days_active)
        )))

        # Generate event timestamps
        timestamps = self._generate_timestamps(
            max(user['account_creation_date'], self.start_date),
            self.end_date,
            num_events,
            user['user_profile']
        )

        # Select devices for events
        devices = user['devices']

        for timestamp in timestamps:
            # Mostly transactions, some logins
            event_type = self.rng.choice(['transaction', 'login'], p=[0.9, 0.1])

            if event_type == 'transaction':
                event = self._generate_transaction(user, timestamp, devices)
            else:
                event = self._generate_login(user, timestamp, devices)

            events.append(event)

        return events

    def _generate_timestamps(
            self,
            start: datetime,
            end: datetime,
            num_events: int,
            profile: str
    ) -> List[datetime]:
        """Generate realistic timestamps for events.

        Args:
            start: Start datetime
            end: End datetime
            num_events: Number of events
            profile: User profile name

        Returns:
            List of timestamps
        """
        # Generate random days
        total_seconds = (end - start).total_seconds()
        random_seconds = self.rng.uniform(0, total_seconds, num_events)
        timestamps = [start + timedelta(seconds=s) for s in sorted(random_seconds)]

        # Adjust hours to be more realistic (peak hours)
        adjusted = []
        for ts in timestamps:
            # Business hours more likely (8-22h)
            if profile == 'corporate':
                hour = int(self.rng.normal(13, 3))  # Peak around 1 PM
                hour = np.clip(hour, 8, 18)
            else:
                # Bimodal: morning (8-10) and evening (18-22)
                if self.rng.random() < 0.5:
                    hour = int(self.rng.normal(9, 1))
                else:
                    hour = int(self.rng.normal(20, 1.5))
                hour = np.clip(hour, 0, 23)

            minute = self.rng.integers(0, 60)
            second = self.rng.integers(0, 60)

            adjusted_ts = ts.replace(hour=hour, minute=minute, second=second)
            adjusted.append(adjusted_ts)

        return adjusted

    def _generate_transaction(
            self,
            user: pd.Series,
            timestamp: datetime,
            devices: List[Dict]
    ) -> Dict[str, Any]:
        """Generate a transaction event.

        Args:
            user: User data
            timestamp: Event timestamp
            devices: User's devices

        Returns:
            Transaction dictionary
        """
        # Select device
        device = self.rng.choice(devices)

        # Select channel
        channels = list(user['channel_probs'].keys())
        probs = list(user['channel_probs'].values())
        channel = self.rng.choice(channels, p=probs)

        # Generate amount
        if user['amount_distribution'] == 'lognormal':
            # Convert mean/std to lognormal parameters
            mean = user['amount_mean']
            std = user['amount_std']
            mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + std ** 2))
            sigma = np.sqrt(np.log(1 + std ** 2 / mean ** 2))
            amount = self.rng.lognormal(mu, sigma)
        else:  # gamma
            shape = (user['amount_mean'] / user['amount_std']) ** 2
            scale = user['amount_std'] ** 2 / user['amount_mean']
            amount = self.rng.gamma(shape, scale)

        amount = round(max(1, amount), 2)

        # Transaction type
        transaction_types = ['wire', 'card', 'internal_transfer', 'bill_payment']
        transaction_type = self.rng.choice(transaction_types)

        # New recipient?
        is_new_recipient = self.rng.random() < user['new_recipient_rate']

        # Generate IP (simplified - will be enhanced by location features module)
        ip_address = self.fake.ipv4()

        # MFA
        mfa_used = self.rng.random() < user['mfa_usage_rate']

        return {
            'event_type': 'transaction',
            'timestamp_utc': timestamp,
            'user_id': user['user_id'],
            'user_profile': user['user_profile'],
            'amount': amount,
            'currency': 'USD',
            'transaction_type': transaction_type,
            'channel': channel,
            'device_id': device['device_id'],
            'device_type': device['device_type'],
            'device_os': device['device_os'],
            'device_os_version': device['device_os_version'],
            'browser_name': device['browser_name'],
            'ip_address': ip_address,
            'is_new_recipient': is_new_recipient,
            'mfa_used': mfa_used,
            'mfa_success': mfa_used,  # Assume success for now
            'is_jailbroken_or_rooted': device['is_jailbroken_or_rooted'],
            'is_emulator_detected': device['is_emulator_detected'],
            # Will be enriched by feature modules
            'registered_country': user['registered_country'],
            'account_creation_date': user['account_creation_date']
        }

    def _generate_login(
            self,
            user: pd.Series,
            timestamp: datetime,
            devices: List[Dict]
    ) -> Dict[str, Any]:
        """Generate a login event.

        Args:
            user: User data
            timestamp: Event timestamp
            devices: User's devices

        Returns:
            Login dictionary
        """
        
        device = self.rng.choice(devices)

        channels = list(user['channel_probs'].keys())
        probs = list(user['channel_probs'].values())
        channel = self.rng.choice(channels, p=probs)

        # Login success rate
        login_success = self.rng.random() < 0.95

        mfa_used = self.rng.random() < user['mfa_usage_rate']

        return {
            'event_type': 'login',
            'timestamp_utc': timestamp,
            'user_id': user['user_id'],
            'user_profile': user['user_profile'],
            'channel': channel,
            'device_id': device['device_id'],
            'device_type': device['device_type'],
            'device_os': device['device_os'],
            'device_os_version': device['device_os_version'],
            'browser_name': device['browser_name'],
            'ip_address': self.fake.ipv4(),
            'login_success': login_success,
            'mfa_used': mfa_used,
            'mfa_success': mfa_used and login_success,
            'is_jailbroken_or_rooted': device['is_jailbroken_or_rooted'],
            'is_emulator_detected': device['is_emulator_detected'],
            'registered_country': user['registered_country'],
            'account_creation_date': user['account_creation_date']
        }