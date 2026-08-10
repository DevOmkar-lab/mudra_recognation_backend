"""Centralized logging setup."""
import logging

def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger without duplicate handlers."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logger
