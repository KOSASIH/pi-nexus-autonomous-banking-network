from typing import List

from blockchain import Blockchain


class BlockRepository:
    """
    A class implementing data access objects (DAOs) for interacting with the blockchain storage.

    Attributes:
        BLOCKCHAIN_FILE (str): The path to the blockchain file.
    """

    BLOCKCHAIN_FILE = "blockchain.json"

    def __init__(self):
        self.blockchain = Blockchain()

    def load_blockchain(self):
        """Load the blockchain from the blockchain file."""
        pass  # Implement loading the blockchain from a file

    def save_blockchain(self):
        """Save the blockchain to the blockchain file."""
        pass  # Implement saving the blockchain to a file

    def add_block(self, data: Any):
        """Add a block to the blockchain."""
        self.blockchain.add_block(data)

    def is_valid(self) -> bool:
        """Check the integrity of the blockchain."""
        return self.blockchain.is_valid()
