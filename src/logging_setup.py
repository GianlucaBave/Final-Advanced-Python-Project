"""Structured logging configuration.

Call :func:`configure_logging` once at process start (the CLI does this). Every
module then obtains its own logger with ``logging.getLogger(__name__)`` and the
records propagate to the shared handlers configured here:

* a console handler at INFO (human-friendly),
* a rotating-by-run file handler at DEBUG written to ``logs/run_<ts>.log``.

The format is ``%(asctime)s | %(levelname)s | %(name)s | %(message)s`` so logs
are greppable and line up in columns.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(
    log_dir: str | Path = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> Path:
    """Configure the root logger. Idempotent — safe to call more than once.

    Returns the path of the run log file so callers can surface it to the user.
    """
    global _CONFIGURED
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Timestamp without microseconds; one file per run for reproducibility.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{run_stamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers do the real filtering

    if _CONFIGURED:
        # Already wired up in this process — just return a fresh file path target.
        return log_path

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence noisy third-party libraries on the console.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured → %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so callers don't import ``logging`` directly."""
    return logging.getLogger(name)
