import logging

def structured_logger(extra):
    """Create a structured logger with custom fields."""
    logger = logging.getLogger('structured_logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('{asctime} {levelname} {message} {extra}', style='{')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger = structured_logger({"app_name": "my_app"})
    logger.debug("This is a debug message", extra={"user_id": 123})
    logger.info("This is an info message", extra={"user_id": 123})
    logger.warning("This is a warning message", extra={"user_id": 123})
    logger.error("This is an error message", extra={"user_id": 123})
    logger.critical("This is a critical message", extra={"user_id": 123})
