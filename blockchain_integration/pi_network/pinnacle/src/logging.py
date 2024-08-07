import logging
import logging.config

class Logger:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "logging.json")
        self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            logging.config.dictConfig(json.load(f))

    def get_logger(self, name):
        return logging.getLogger(name)

logger = Logger()
