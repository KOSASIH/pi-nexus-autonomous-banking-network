import logging


class Logging:
    def __init__(self):
        self.logger = logging.getLogger("banking_network")
        self.logger.setLevel(logging.DEBUG)

    def setup_logging(self):
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, message):
        self.logger.info(message)
