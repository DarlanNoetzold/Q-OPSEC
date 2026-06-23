from __future__ import annotations

import argparse
from pathlib import Path

from src.common.logger import get_logger
from src.dataset_generation.orchestrator import DatasetOrchestrator

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="Generate fraud detection dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated dataset"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    config_path = args.config
    output_dir = args.output_dir

    logger.info(f"Using config: {config_path}")
    logger.info(f"Output directory: {output_dir}")

    orchestrator = DatasetOrchestrator(
        config_path=config_path,
        output_dir=output_dir
    )
    orchestrator.run()

    logger.info("=" * 80)
    logger.info("✅ Dataset generation completed successfully!")
    logger.info(f"📁 Output saved to: {Path(output_dir).absolute()}")
    logger.info("=" * 80)
    logger.info("\n📊 Next steps:")
    logger.info("  1. Explore the dataset: output/dataset_summary.txt")
    logger.info("  2. Train models: python train_model.py --data output/")
    logger.info("  3. Deploy API: python api_server.py --model models/best_model.pkl")


if __name__ == "__main__":
    main()