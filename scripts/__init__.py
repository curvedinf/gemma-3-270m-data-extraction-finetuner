"""
Helper modules backing Fabric tasks.

Each module contains pure-Python functions so they can be unit tested without
invoking Fabric contexts directly.
"""

import logging


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("gemma_finetune")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOGGER = _configure_logging()
