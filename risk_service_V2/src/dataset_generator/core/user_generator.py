import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from faker import Faker
from src.common.logger import log


class UserGenerator:
    def __init__(self, config: Dict[str, Any], random_seed: int = 42):
        """Initialize user generator.

        Args:
            config: User profiles configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.profiles = config['profiles']
        self.device_chars = config['device_characteristics']

        self.rng = np.random.default_rng(random_seed)
        self.fake = Faker()
        Faker.seed(random_seed)


    def generate_users(self, num_users: int, start_date: datetime) -> pd.DataFrame:
        """Generate user dataset.

        Args:
            num_users: Number of users to generate
            start_date: Start date for account creation

        Returns:
            DataFrame with user data
        """
        log.info(f"Generating {num_users} users...")

        users = []

        # Determine profile distribution
        profile_names = list(self.profiles.keys())
        profile_weights = [self.profiles[p]['weight'] for p in profile_names]

        for user_idx in range(num_users):
            # Select profile
            profile_name = self.rng.choice(profile_names, p=profile_weights)
            profile = self.profiles[profile_name]

            # Generate user data
            user = self._generate_single_user(
                user_idx,
                profile_name,
                profile,
                start_date
            )
            users.append(user)

        df = pd.DataFrame(users)
        log.info(f"✓ Generated {len(df)} users")
        log.info(f"  Profile distribution:\n{df['user_profile'].value_counts()}")

        return df


    def _generate_single_user(
            self,
            user_idx: int,
            profile_name: str,
            profile: Dict[str, Any],
            start_date: datetime
    ) -> Dict[str, Any]:
        """Generate a single user.

        Args:
            user_idx: User index
            profile_name: Name of user profile
            profile: Profile configuration
            start_date: Dataset start date

        Returns:
            Dictionary with user data
        """
        # Account creation date (random within 1-3 years before start_date)
        days_before = self.rng.integers(30, 1095)  # 1 month to 3 years
        account_creation = start_date - timedelta(days=days_before)

        # Select primary country
        country = self.rng.choice(profile['countries'])

        # Generate devices
        device_chars = self.device_chars[profile_name]
        num_devices = max(1, int(self.rng.normal(
            device_chars['devices_per_user_mean'],
            device_chars['devices_per_user_std']
        )))

        devices = self._generate_devices(num_devices, device_chars)

        return {
            'user_id': f"user_{user_idx:08d}",
            'user_profile': profile_name,
            'user_type': 'business' if profile_name == 'corporate' else 'individual',
            'user_segment': profile_name,
            'account_creation_date': account_creation,
            'registered_country': country,
            'registered_region': self.fake.state(),
            'transactions_per_day_mean': profile['transactions_per_day_mean'],
            'transactions_per_day_std': profile['transactions_per_day_std'],
            'amount_mean': profile['amount_mean'],
            'amount_std': profile['amount_std'],
            'amount_distribution': profile['amount_distribution'],
            'channel_probs': profile['channels'],
            'mfa_usage_rate': profile['mfa_usage_rate'],
            'new_recipient_rate': profile['new_recipient_rate'],
            'devices': devices,
            'num_devices': len(devices),
            'is_fraudster': profile_name == 'fraudster'
        }


    def _generate_devices(
            self,
            num_devices: int,
            device_chars: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate devices for a user.

        Args:
            num_devices: Number of devices
            device_chars: Device characteristics config

        Returns:
            List of device dictionaries
        """
        devices = []

        os_names = list(device_chars['os_distribution'].keys())
        os_probs = list(device_chars['os_distribution'].values())

        for dev_idx in range(num_devices):
            os_name = self.rng.choice(os_names, p=os_probs)

            # Determine if outdated
            is_outdated = self.rng.random() < device_chars['outdated_os_rate']

            device = {
                'device_id': self.fake.uuid4(),
                'device_type': self._get_device_type(os_name),
                'device_os': os_name,
                'device_os_version': self._get_os_version(os_name, is_outdated),
                'browser_name': self._get_browser(os_name),
                'is_primary': dev_idx == 0,
                'is_jailbroken_or_rooted': self.rng.random() < 0.05,
                'is_emulator_detected': self.rng.random() < device_chars.get('emulator_rate', 0.02)
            }

            devices.append(device)

        return devices


    def _get_device_type(self, os_name: str) -> str:
        """Get device type from OS."""
        if os_name in ['Android', 'iOS']:
            return 'mobile'
        elif os_name in ['Windows', 'macOS', 'Linux']:
            return 'desktop'
        return 'other'


    def _get_os_version(self, os_name: str, is_outdated: bool) -> str:
        """Get OS version."""
        versions = {
            'Android': ['14', '13', '12', '11', '10', '9'] if not is_outdated else ['8', '7', '6'],
            'iOS': ['17', '16', '15', '14'] if not is_outdated else ['13', '12', '11'],
            'Windows': ['11', '10'] if not is_outdated else ['8.1', '7'],
            'macOS': ['14', '13', '12'] if not is_outdated else ['11', '10.15'],
            'Linux': ['6.x', '5.x']
        }

        return self.rng.choice(versions.get(os_name, ['unknown']))


    def _get_browser(self, os_name: str) -> str:
        """Get browser based on OS."""
        if os_name == 'iOS':
            return self.rng.choice(['Safari', 'Chrome'], p=[0.7, 0.3])
        elif os_name == 'Android':
            return self.rng.choice(['Chrome', 'Firefox', 'Samsung Internet'], p=[0.7, 0.2, 0.1])
        else:
            return self.rng.choice(['Chrome', 'Firefox', 'Edge', 'Safari'], p=[0.5, 0.2, 0.2, 0.1])
