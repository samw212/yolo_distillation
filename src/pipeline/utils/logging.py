"""Logging setup for the distillation pipeline."""

import logging
import sys
from pathlib import Path


def setup_logging(config: dict) -> logging.Logger:
    """Configure logging based on pipeline config.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Configured root logger.
    """
    pipeline_cfg = config.get("pipeline", {})
    log_level = getattr(logging, pipeline_cfg.get("log_level", "INFO").upper())
    log_file = pipeline_cfg.get("log_file")

    logger = logging.getLogger("distill")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
