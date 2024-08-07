import logging
import logging.config
import os

class Logger:
    def __init__(self):
        self.config = config.get('logging_level')
        self.setup_logging()

    def setup_logging(self):
        logging.config.dictConfig({
            'version': 1,
            'formatters': {
                'default': {
                    'format': '[%(asctime)s] %(levelname)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.config,
                    'formatter': 'default'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': 'app.log',
                    'level': self.config,
                    'formatter': 'default'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': self.config
                }
            }
        })

    def get_logger(self, name):
        return logging.getLogger(name)

logger = Logger()
