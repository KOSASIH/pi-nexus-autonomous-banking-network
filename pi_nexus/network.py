# pi_nexus/network.py
import logging

logger = logging.getLogger(__name__)

class Network:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self) -> None:
        try:
            self.socket.connect((self.host, self.port))
        except socket.error as e:
            logger.error(f"Error connecting to {self.host}:{self.port}: {e}")
            raise
