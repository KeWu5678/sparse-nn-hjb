"""Run with ``uv run --frozen python -m scripts.logging_demo --log-file /tmp/demo.jsonl``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import structlog

from src.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Show console and JSON Lines logging.")
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    configure_logging(log_file=args.log_file, level=args.level)

    logger = structlog.get_logger(__name__).bind(example="logging")
    logger.debug("debug detail", enabled_with="--level DEBUG")
    logger.info("demo started", items=3)
    logging.getLogger(__name__).info("standard logging also works", extra={"items": 3})
    try:
        raise ValueError("example failure")
    except ValueError:
        logger.exception("demo exception", handled=True)
    logger.info("demo finished", log_file=str(args.log_file))


if __name__ == "__main__":
    main()
