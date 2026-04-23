from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.config_loader import load_config
from src.core.csv_handler import CsvHandler
from src.core.prompt_builder import build_user_prompt
from src.core.runner import run
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Capability Evaluation Framework")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete existing checkpoint before execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and CSV, preview first 3 prompts, no API calls and no file writes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("data/output/run.log", level=logging.INFO)

    config = load_config(args.config)

    checkpoint_path = Path(config.execution.checkpoint_path)
    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed due to --reset-checkpoint: %s", checkpoint_path)

    csv_handler = CsvHandler(config)
    df = csv_handler.load()

    if args.dry_run:
        logger.info("Running in dry-run mode. Validating first 3 prompts.")
        max_rows = min(3, len(df.index))
        for i in range(max_rows):
            prompt = build_user_prompt(df.iloc[i], config)
            logger.info("Dry-run prompt[%d]: %s", i, prompt)
        logger.info("Dry-run completed. No API calls, no file modifications.")
        return

    run(config)


if __name__ == "__main__":
    main()
