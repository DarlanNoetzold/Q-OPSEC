import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Any
from src.common.logger import log


class TemporalFeatureExtractor:
    """Extracts temporal and time-based behavioral features."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize temporal feature extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def extract(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features for all events.

        Args:
            events_df: DataFrame with events

        Returns:
            DataFrame with added temporal features
        """
        log.info("Extracting temporal features...")

        df = events_df.copy()

        # Ensure timestamp is datetime
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

        # Basic time features
        df['hour_of_day'] = df['timestamp_utc'].dt.hour
        df['day_of_week'] = df['timestamp_utc'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # TODO: Add holiday detection (would need holiday calendar)
        df['is_local_holiday'] = 0

        # Sort by user and time for sequential features
        df = df.sort_values(['user_id', 'timestamp_utc']).reset_index(drop=True)

        # Time since last event (per user)
        df['seconds_since_last_event'] = df.groupby('user_id')['timestamp_utc'].diff().dt.total_seconds()
        df['seconds_since_last_event'] = df['seconds_since_last_event'].fillna(0)

        # For transactions only
        transaction_mask = df['event_type'] == 'transaction'

        if transaction_mask.any():
            # Time since last transaction
            df.loc[transaction_mask, 'seconds_since_last_transaction'] = (
                df[transaction_mask].groupby('user_id')['timestamp_utc']
                .diff().dt.total_seconds()
            )
            df['seconds_since_last_transaction'] = df['seconds_since_last_transaction'].fillna(0)
        else:
            df['seconds_since_last_transaction'] = 0

        # Rolling window features
        df = self._add_rolling_features(df)

        # Account age at time of event
        df['account_age_days'] = (
                df['timestamp_utc'] - pd.to_datetime(df['account_creation_date'])
        ).dt.days

        log.info(f"✓ Added temporal features")

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window aggregation features.

        Args:
            df: DataFrame with events

        Returns:
            DataFrame with rolling features
        """
        # We'll compute these for each event by looking back in time
        # This is computationally expensive for large datasets
        # In production, you'd use a more efficient approach (e.g., streaming)

        windows = {
            '1h': timedelta(hours=1),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }

        # Initialize columns
        for window_name in windows.keys():
            df[f'transactions_last_{window_name}'] = 0
            df[f'amount_sum_last_{window_name}'] = 0.0
            df[f'logins_last_{window_name}'] = 0
            df[f'login_failures_last_{window_name}'] = 0

        # Group by user for efficiency
        for user_id, user_events in df.groupby('user_id'):
            user_indices = user_events.index

            for idx in user_indices:
                current_time = df.loc[idx, 'timestamp_utc']

                # Get all events before current event for this user
                prior_events = user_events[user_events['timestamp_utc'] < current_time]

                if len(prior_events) == 0:
                    continue

                for window_name, window_delta in windows.items():
                    window_start = current_time - window_delta
                    window_events = prior_events[prior_events['timestamp_utc'] >= window_start]

                    # Count transactions
                    transactions = window_events[window_events['event_type'] == 'transaction']
                    df.loc[idx, f'transactions_last_{window_name}'] = len(transactions)

                    # Sum amounts
                    if len(transactions) > 0 and 'amount' in transactions.columns:
                        df.loc[idx, f'amount_sum_last_{window_name}'] = transactions['amount'].sum()

                    # Count logins
                    logins = window_events[window_events['event_type'] == 'login']
                    df.loc[idx, f'logins_last_{window_name}'] = len(logins)

                    # Count login failures
                    if len(logins) > 0 and 'login_success' in logins.columns:
                        failures = logins[logins['login_success'] == False]
                        df.loc[idx, f'login_failures_last_{window_name}'] = len(failures)

        # Compute mean and std for 30d window (for anomaly detection)
        df['amount_mean_last_30d'] = 0.0
        df['amount_std_last_30d'] = 0.0

        for user_id, user_events in df.groupby('user_id'):
            user_indices = user_events.index

            for idx in user_indices:
                current_time = df.loc[idx, 'timestamp_utc']
                window_start = current_time - timedelta(days=30)

                prior_transactions = user_events[
                    (user_events['timestamp_utc'] >= window_start) &
                    (user_events['timestamp_utc'] < current_time) &
                    (user_events['event_type'] == 'transaction')
                    ]

                if len(prior_transactions) > 0 and 'amount' in prior_transactions.columns:
                    amounts = prior_transactions['amount']
                    df.loc[idx, 'amount_mean_last_30d'] = amounts.mean()
                    if len(amounts) > 1:
                        df.loc[idx, 'amount_std_last_30d'] = amounts.std()

        return df