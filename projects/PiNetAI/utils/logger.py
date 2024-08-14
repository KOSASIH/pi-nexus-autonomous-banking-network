# logger.py

import logging

# Create a custom logger
logger = logging.getLogger("my_app")

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add the formatter to the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_debug(message):
    logger.debug(message)

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_critical(message):
    logger.critical(message)
