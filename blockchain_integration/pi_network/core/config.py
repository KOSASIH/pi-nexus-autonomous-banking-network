# config.py
import os


class Config:
    def __init__(self):
        self.pi_network_port = int(os.environ.get("PI_NETWORK_PORT", 8080))
        self.pi_network_host = os.environ.get("PI_NETWORK_HOST", "localhost")
        self.blockchain_node_url = os.environ.get(
            "BLOCKCHAIN_NODE_URL", "https://example.com/blockchain"
        )
