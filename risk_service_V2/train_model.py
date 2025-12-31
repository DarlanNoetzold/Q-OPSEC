"""
Train fraud detection models
"""
import argparse
from pathlib import Path
import yaml

from src.model_training.trainer import ModelTrainer
from src.common.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")

    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML config file as dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration (direto, sem ConfigLoader)
    config = load_config(args.config)

    logger.info("=" * 80)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Log level: {args.log_level}")

    try:
        # Initialize trainer
        trainer = ModelTrainer(config)

        # Run training pipeline
        trainer.train_pipeline()

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()