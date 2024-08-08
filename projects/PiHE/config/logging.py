import logging
import os

# Set up logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename=os.path.join(os.path.dirname(__file__), 'pihe.log'),
    filemode='w'
)

# Create a logger for the PiHE project
logger = logging.getLogger('pihe')

# Set up logging levels for different modules
logging.getLogger('crypto').setLevel(logging.DEBUG)
logging.getLogger('models').setLevel(logging.INFO)
logging.getLogger('applications').setLevel(logging.INFO)

# Define a custom logging handler for console output
class ConsoleHandler(logging.StreamHandler):
    def __init__(self):
        super(ConsoleHandler, self).__init__()

    def emit(self, record):
        msg = self.format(record)
        print(msg)

# Add the custom console handler to the logger
console_handler = ConsoleHandler()
logger.addHandler(console_handler)

# Define a custom logging handler for file output
class FileHandler(logging.FileHandler):
    def __init__(self, filename):
        super(FileHandler, self).__init__(filename)

    def emit(self, record):
        msg = self.format(record)
        self.stream.write(msg + '\n')

# Add the custom file handler to the logger
file_handler = FileHandler(os.path.join(os.path.dirname(__file__), 'pihe.log'))
logger.addHandler(file_handler)

# Log a message to test the logging configuration
logger.info('PiHE logging system initialized')
