"""Consistent console and per-run file logging for CFSurv."""

from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "cfsurv"
FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def configure_run_logging(log_path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_cfsurv_file_handler", False):
            logger.removeHandler(handler)
            handler.close()

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not getattr(handler, "_cfsurv_file_handler", False)
        for handler in logger.handlers
    ):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(stream)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FORMAT))
    file_handler._cfsurv_file_handler = True
    logger.addHandler(file_handler)
    return logger


def get_run_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
