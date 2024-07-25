# dex_project_logger.py
import logging

class DexProjectLogger:
    def __init__(self):
        self.logger = logging.getLogger('dex_project_logger')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_info(self, message):
        # Log info message
        self.logger.info(message)

    def log_error(self, message):
        # Log error message
        self.logger.error(message)
