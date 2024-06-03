# logger.py
import logging


class Logger:
    def __init__(self):
        self.logger = logging.getLogger("pi_network")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
