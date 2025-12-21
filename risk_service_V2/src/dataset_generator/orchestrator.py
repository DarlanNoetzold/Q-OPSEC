import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm

from src.common.config_loader import ConfigLoader
from src.common.logger import log
from src.dataset_generation.core.user_generator import UserGenerator
from src.dataset_generation.core.event_generator import EventGenerator
from src.dataset_generation.features.temporal_features import TemporalFeatureExtractor
from src.dataset_generation.utils.ollama_client import OllamaClient


class DatasetOrchestrator:
    """Orchestrates the entire dataset generation pipeline."""

    def __init__(self, config_dir: str = "config"):
        """Initialize orchestrator.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.configs = self.config_loader.load_all()

        self.dataset_config = self.configs['dataset_config']
        self.user_profiles_config = self.configs['user_profiles']
        self.fraud_scenarios_config = self.configs['fraud_scenarios']
        self.llm_config = self.configs['llm_config']

        self.random_seed = self.dataset_config['generation']['random_seed']

        # Initialize components
        self.user_generator = None
        self.event_generator = None
        self.feature_extractors = {}
        self.llm_client = None

        # Data holders
        self.users_df = None
        self.events_df = None
        self.final_dataset = None

    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset.

        Returns:
            Final dataset DataFrame
        """
        log.info("=" * 80)
        log.info("STARTING DATASET GENERATION")
        log.info("=" * 80)

        start_time = datetime.now()

        # Step 1: Generate users
        self._generate_users()

        # Step 2: Generate events
        self._generate_events()

        # Step 3: Extract features
        self._extract_features()

        # Step 4: Add LLM features (if enabled)
        if self.dataset_config['features']['llm']['enabled']:
            self._add_llm_features()

        # Step 5: Add labels
        self._add_labels()

        # Step 6: Save dataset
        self._save_dataset()

        elapsed = (datetime.now() - start_time).total_seconds()
        log.info("=" * 80)
        log.info(f"✓ DATASET GENERATION COMPLETE in {elapsed:.1f}s")
        log.info(f"  Total events: {len(self.final_dataset)}")
        log.info(f"  Total features: {len(self.final_dataset.columns)}")
        log.info("=" * 80)

        return self.final_dataset

    def _generate_users(self):
        """Generate user dataset."""
        log.info("\n" + "=" * 80)
        log.info("STEP 1: Generating Users")
        log.info("=" * 80)

        self.user_generator = UserGenerator(
            self.user_profiles_config,
            random_seed=self.random_seed
        )

        num_users = self.dataset_config['generation']['num_users']
        start_date = datetime.strptime(
            self.dataset_config['generation']['start_date'],
            '%Y-%m-%d'
        )

        self.users_df = self.user_generator.generate_users(num_users, start_date)

    def _generate_events(self):
        """Generate events for all users."""
        log.info("\n" + "=" * 80)
        log.info("STEP 2: Generating Events")
        log.info("=" * 80)

        self.event_generator = EventGenerator(
            self.users_df,
            self.dataset_config,
            random_seed=self.random_seed
        )

        self.events_df = self.event_generator.generate_events()

    def _extract_features(self):
        """Extract all features."""
        log.info("\n" + "=" * 80)
        log.info("STEP 3: Extracting Features")
        log.info("=" * 80)

        self.final_dataset = self.events_df.copy()

        # Get enabled feature modules
        enabled_modules = self.dataset_config['features']['enabled_modules']

        # Temporal features
        if 'temporal_features' in enabled_modules:
            temporal_extractor = TemporalFeatureExtractor(self.dataset_config)
            self.final_dataset = temporal_extractor.extract(self.final_dataset)

        # TODO: Add other feature extractors
        # - location_features
        # - device_features
        # - security_features
        # - transaction_features
        # - behavioral_features
        # - text_features

        log.info(f"✓ Feature extraction complete. Total columns: {len(self.final_dataset.columns)}")

    def _add_llm_features(self):
        """Add LLM-generated features."""
        log.info("\n" + "=" * 80)
        log.info("STEP 4: Adding LLM Features")
        log.info("=" * 80)

        # Initialize LLM client
        self.llm_client = OllamaClient(self.llm_config)

        sample_rate = self.dataset_config['features']['llm']['sample_rate']
        batch_size = self.dataset_config['features']['llm']['batch_size']

        # Sample events to process
        if sample_rate < 1.0:
            sample_indices = self.final_dataset.sample(
                frac=sample_rate,
                random_state=self.random_seed
            ).index
        else:
            sample_indices = self.final_dataset.index

        log.info(f"Processing {len(sample_indices)} events with LLM (sample rate: {sample_rate})")

        # Initialize LLM feature columns
        llm_columns = [
            'llm_risk_score', 'llm_risk_level', 'llm_risk_category',
            'llm_fraud_pattern', 'llm_detected_social_engineering',
            'llm_detected_urgency', 'llm_detected_suspicious_link',
            'llm_short_explanation', 'llm_risk_tags',
            'llm_model_name', 'llm_model_version', 'llm_prompt_version'
        ]

        for col in llm_columns:
            if col == 'llm_risk_tags':
                self.final_dataset[col] = [[] for _ in range(len(self.final_dataset))]
            elif col in ['llm_risk_score']:
                self.final_dataset[col] = 0.5
            elif col in ['llm_detected_social_engineering', 'llm_detected_urgency', 'llm_detected_suspicious_link']:
                self.final_dataset[col] = False
            else:
                self.final_dataset[col] = ''

        # Process in batches with progress bar
        for i in tqdm(range(0, len(sample_indices), batch_size), desc="LLM processing"):
            batch_indices = sample_indices[i:i + batch_size]

            for idx in batch_indices:
                row = self.final_dataset.loc[idx]

                # Prepare transaction data for LLM
                transaction_data = {
                    'amount': row.get('amount', 0),
                    'currency': row.get('currency', 'USD'),
                    'transaction_type': row.get('transaction_type', 'unknown'),
                    'channel': row.get('channel', 'unknown'),
                    'account_age_days': row.get('account_age_days', 0),
                    'is_new_recipient': row.get('is_new_recipient', False),
                    'country_change_since_last_session': False,  # TODO: compute this
                    'is_new_device': False,  # TODO: compute this
                    'mfa_used': row.get('mfa_used', False),
                    'hour_of_day': row.get('hour_of_day', 12),
                    'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][row.get('day_of_week', 0)],
                    'message_text': row.get('message_text', 'N/A')
                }

                # Get LLM assessment
                llm_result = self.llm_client.assess_risk(transaction_data)

                # Update dataframe
                for key, value in llm_result.items():
                    if key in llm_columns:
                        self.final_dataset.at[idx, key] = value

        log.info("✓ LLM features added")

    def _add_labels(self):
        """Add ground truth labels."""
        log.info("\n" + "=" * 80)
        log.info("STEP 5: Adding Labels")
        log.info("=" * 80)

        # Simple rule-based labeling for now
        # In a real scenario, you'd use fraud_injector and more sophisticated labeling

        # Initialize label columns
        self.final_dataset['label_fraud'] = 0
        self.final_dataset['label_risk_level'] = 0  # 0=low, 1=medium, 2=high, 3=critical
        self.final_dataset['label_source'] = 'rule_based'
        self.final_dataset['label_confidence'] = 1.0

        # Mark fraudster profile events as fraud
        fraud_mask = self.final_dataset['user_profile'] == 'fraudster'
        self.final_dataset.loc[fraud_mask, 'label_fraud'] = 1
        self.final_dataset.loc[fraud_mask, 'label_risk_level'] = 3

        # Add some medium risk cases
        medium_risk_mask = (
                (self.final_dataset['amount'] > 1000) &
                (self.final_dataset['is_new_recipient'] == True) &
                (self.final_dataset['label_fraud'] == 0)
        )
        self.final_dataset.loc[medium_risk_mask, 'label_risk_level'] = 1

        fraud_rate = self.final_dataset['label_fraud'].mean()
        log.info(f"✓ Labels added. Fraud rate: {fraud_rate:.2%}")

    def _save_dataset(self):
        """Save final dataset to disk."""
        log.info("\n" + "=" * 80)
        log.info("STEP 6: Saving Dataset")
        log.info("=" * 80)

        output_dir = Path(self.dataset_config['dataset']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = self.dataset_config['dataset']['name']
        output_format = self.dataset_config['dataset']['output_format']

        # Save full dataset
        if output_format in ['parquet', 'both']:
            output_path = output_dir / f"{dataset_name}.parquet"
            self.final_dataset.to_parquet(output_path, index=False)
            log.info(f"✓ Saved to {output_path}")

        if output_format in ['csv', 'both']:
            output_path = output_dir / f"{dataset_name}.csv"
            self.final_dataset.to_csv(output_path, index=False)
            log.info(f"✓ Saved to {output_path}")

        # Save train/val/test splits if enabled
        if self.dataset_config['export']['split_train_val_test']:
            self._save_splits(output_dir, dataset_name)

        # Save metadata
        self._save_metadata(output_dir, dataset_name)

    def _save_splits(self, output_dir: Path, dataset_name: str):
        """Save train/validation/test splits."""
        log.info("Creating train/val/test splits...")

        if self.dataset_config['export']['time_based_split']:
            # Time-based split
            df_sorted = self.final_dataset.sort_values('timestamp_utc')

            train_ratio = self.dataset_config['export']['train_ratio']
            val_ratio = self.dataset_config['export']['val_ratio']

            n = len(df_sorted)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_df = df_sorted.iloc[:train_end]
            val_df = df_sorted.iloc[train_end:val_end]
            test_df = df_sorted.iloc[val_end:]
        else:
            # Random split
            train_df = self.final_dataset.sample(
                frac=self.dataset_config['export']['train_ratio'],
                random_state=self.random_seed
            )
            remaining = self.final_dataset.drop(train_df.index)

            val_frac = self.dataset_config['export']['val_ratio'] / (
                    self.dataset_config['export']['val_ratio'] + self.dataset_config['export']['test_ratio']
            )
            val_df = remaining.sample(frac=val_frac, random_state=self.random_seed)
            test_df = remaining.drop(val_df.index)

        # Save splits
        train_df.to_parquet(output_dir / f"{dataset_name}_train.parquet", index=False)
        val_df.to_parquet(output_dir / f"{dataset_name}_val.parquet", index=False)
        test_df.to_parquet(output_dir / f"{dataset_name}_test.parquet", index=False)

        log.info(f"  Train: {len(train_df)} ({len(train_df) / len(self.final_dataset):.1%})")
        log.info(f"  Val:   {len(val_df)} ({len(val_df) / len(self.final_dataset):.1%})")
        log.info(f"  Test:  {len(test_df)} ({len(test_df) / len(self.final_dataset):.1%})")

    def _save_metadata(self, output_dir: Path, dataset_name: str):
        """Save dataset metadata."""
        metadata = {
            'dataset_name': dataset_name,
            'generation_date': datetime.now().isoformat(),
            'num_events': len(self.final_dataset),
            'num_users': self.final_dataset['user_id'].nunique(),
            'num_features': len(self.final_dataset.columns),
            'fraud_rate': float(self.final_dataset['label_fraud'].mean()),
            'time_period': {
                'start': str(self.final_dataset['timestamp_utc'].min()),
                'end': str(self.final_dataset['timestamp_utc'].max())
            },
            'config': self.dataset_config
        }

        import json
        with open(output_dir / f"{dataset_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        log.info(f"✓ Metadata saved")