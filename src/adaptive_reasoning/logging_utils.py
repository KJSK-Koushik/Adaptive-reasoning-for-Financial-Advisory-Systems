"""Console + file logging, shared by every script."""

from __future__ import annotations

import logging
from datetime import datetime

from . import paths

_CONFIGURED = False


def setup_logging(level: str = "INFO", to_file: bool = True, rich_console: bool = True,
                  run_name: str | None = None) -> logging.Logger:
    """Configure root logging once. Safe to call from any entry point."""
    global _CONFIGURED
    logger = logging.getLogger("adaptive_reasoning")
    if _CONFIGURED:
        return logger

    logger.setLevel(level.upper())
    logger.propagate = False

    if rich_console:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(handler)

    if to_file:
        paths.LOGS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{run_name}-{stamp}.log" if run_name else f"run-{stamp}.log"
        fh = logging.FileHandler(paths.LOGS / name, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"adaptive_reasoning.{name}" if name else "adaptive_reasoning")
