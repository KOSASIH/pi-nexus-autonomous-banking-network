import logging
from config import get_config

def setup_logging():
    # Set up logging to console and file
    log_level = get_config('logging.log_level')
    log_file = get_config('logging.log_file')

    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s', filename=log_file, filemode='w')

    # Create a logger for the application
    logger = logging.getLogger(__name__)

    return logger
