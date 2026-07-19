"""
retraining_logger.py
Structured logger setup for Phase 12: Automatic Model Retraining System.
"""
import logging
import os
from datetime import datetime
from typing import Optional

_LOGGER_NAME = "retraining_pipeline"
_handler_initialized: bool = False


def setup_logger(
    logs_dir: str,
    log_filename: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configures and returns the retraining pipeline logger.

    Creates the *logs_dir* if necessary and attaches both a rotating
    file handler and a console handler (INFO+) to the named logger so
    that the same logger object is returned on repeated calls without
    adding duplicate handlers.

    Args:
        logs_dir:     Directory where log files will be written.
        log_filename: Optional explicit filename; defaults to
                      ``retraining_pipeline_<timestamp>.log``.
        level:        Root log level (default: DEBUG).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    global _handler_initialized  # noqa: PLW0603

    logger = logging.getLogger(_LOGGER_NAME)

    if _handler_initialized:
        return logger

    logger.setLevel(level)

    # Ensure log directory exists
    os.makedirs(logs_dir, exist_ok=True)

    if log_filename is None:
        log_filename = (
            f"retraining_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    log_path = os.path.join(logs_dir, log_filename)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler – DEBUG+
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except OSError:
                pass

    # Console handler – INFO+
    ch = SafeStreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _handler_initialized = True
    logger.info("Retraining logger initialized. Log file: %s", log_path)
    return logger


def shutdown_logger() -> None:
    """Flushes and closes all handlers attached to the retraining logger."""
    global _handler_initialized  # noqa: PLW0603
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    _handler_initialized = False


def get_logger() -> logging.Logger:
    """Returns the shared retraining logger (must call :func:`setup_logger` first)."""
    return logging.getLogger(_LOGGER_NAME)
