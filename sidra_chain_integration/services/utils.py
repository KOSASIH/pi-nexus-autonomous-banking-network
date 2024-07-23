import logging

logger = logging.getLogger(__name__)


def log_error(message: str):
    logger.error(message)
