import logging


class HITLInterface:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def display_status(self):
        """Design a user-friendly interface for human operators to monitor and control the ASRS."""
        self.logger.info("Displaying status...")
        # Implement status display logic here

    def receive_operator_input(self):
        """Enable operators to intervene and override automated decisions when necessary."""
        self.logger.info("Receiving operator input...")
        # Implement operator input logic here
