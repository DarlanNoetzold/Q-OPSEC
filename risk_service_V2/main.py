import argparse
from pathlib import Path

from src.common.logger import setup_logger, log
from src.dataset_generation.orchestrator import DatasetOrchestrator


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fraud Risk Dataset Generator"
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Directory containing configuration files'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_dir=args.log_dir, log_level=args.log_level)

    log.info("=" * 80)
    log.info("FRAUD RISK DATASET GENERATOR")
    log.info("=" * 80)
    log.info(f"Config directory: {args.config_dir}")
    log.info(f"Log level: {args.log_level}")
    log.info("")

    # Create orchestrator and generate dataset
    try:
        orchestrator = DatasetOrchestrator(config_dir=args.config_dir)
        dataset = orchestrator.generate_dataset()

        log.info("")
        log.info("=" * 80)
        log.info("SUCCESS!")
        log.info("=" * 80)
        log.info(f"Dataset shape: {dataset.shape}")
        log.info(f"Columns: {list(dataset.columns)}")
        log.info("")
        log.info("Next steps:")
        log.info("  1. Explore the dataset: jupyter notebook notebooks/01_explore_dataset.ipynb")
        log.info("  2. Train models: python -m src.model_training.trainer")
        log.info("  3. Start API: python -m src.api.app")

    except Exception as e:
        log.error(f"Dataset generation failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return 1

    return 0


if __name__ == '__main__':
    exit(main())