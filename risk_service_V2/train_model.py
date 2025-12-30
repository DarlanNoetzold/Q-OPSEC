import argparse
from pathlib import Path

from src.common.config_loader import ConfigLoader
from src.common.logger import logger
from src.model_training.trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "training.log",
        level=args.log_level,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}",
    )

    logger.info("=" * 80)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Log level: {args.log_level}")

    # Load configuration
    config_path = Path(args.config)
    config_loader = ConfigLoader(config_path.parent)

    config = config_loader.load(config_path.name)

    # Initialize trainer
    trainer = ModelTrainer(config)

    try:
        trainer.run()
        logger.info("\n✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()