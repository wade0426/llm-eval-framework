from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_file_path: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("llm_eval")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
