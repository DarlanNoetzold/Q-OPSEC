from __future__ import annotations

import argparse
from pathlib import Path

from src.common.logger import get_logger
from src.dataset_generation.orchestrator import DatasetOrchestrator


logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic Digital Security Dataset Generator")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset configuration YAML (relative to config/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write generated datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config
    # Allow passing either full path or just name under config/
    if not Path(config_path).exists():
        config_path = str(config_path)

    output_dir = Path(args.output_dir)
    logger.info("Using config: {cfg}", cfg=config_path)
    logger.info("Output directory: {out}", out=str(output_dir))

    orchestrator = DatasetOrchestrator(output_dir=output_dir, dataset_config_path=config_path)
    orchestrator.run()


if __name__ == "__main__":
    main()
