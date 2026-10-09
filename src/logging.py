"""Logging setup for entry points using structlog or stdlib logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(
    *, log_file: str | Path, level: int | str = "INFO", console: bool = True,
) -> None:
    """Replace root handlers with JSON Lines output and optional readable stderr.

    Both outputs use ``level``; ``console=False`` keeps file output.
    Call at the entry point, then use
    ``structlog.get_logger(__name__)`` or ordinary ``logging.getLogger`` calls.
    Reconfiguration closes the previous root handlers and changes both outputs.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    foreign_processors = [structlog.stdlib.ExtraAdder(), *shared_processors]
    remove_metadata = structlog.stdlib.ProcessorFormatter.remove_processors_meta

    handlers = []
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_processors,
            processors=[remove_metadata, structlog.dev.ConsoleRenderer(
                colors=False, exception_formatter=structlog.dev.plain_traceback,
            )],
        ))
        handlers.append(console_handler)
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=foreign_processors,
        processors=[remove_metadata, structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer()],
    ))
    handlers.append(file_handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
