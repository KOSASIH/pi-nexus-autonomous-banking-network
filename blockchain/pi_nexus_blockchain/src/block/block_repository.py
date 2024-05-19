import json
from pathlib import Path

from block_model import Block


class BlockRepository:
    """
    A class implementing data access objects (DAOs) for interacting with the blockchain storage.

    Attributes:
        BLOCKCHAIN_FILE (Path): The path to the blockchain file.
    """

    BLOCKCHAIN_FILE = Path("blockchain.json")

    def __init__(self):
        if not self.BLOCKCHAIN_FILE.exists():
            self.BLOCKCHAIN_FILE.touch()

    def load_blockchain(self) -> list[Block]:
        """Load the blockchain from the blockchain file."""
        if not self.BLOCKCHAIN_FILE.exists():
            return []

        with open(self.BLOCKCHAIN_FILE, "r") as f:
            blockchain = json.load(f)

        return [Block.from_dict(block) for block in blockchain]

    def save_blockchain(self, blockchain: list[Block]):
        """Save the blockchain to the blockchain file."""
        with open(self.BLOCKCHAIN_FILE, "w") as f:
            json.dump([block.to_dict() for block in blockchain], f)

    def add_block(self, block: Block):
        """Add a block to the blockchain."""
        blockchain = self.load_blockchain()
        blockchain.append(block)
        self.save_blockchain(blockchain)

    def get_latest_block(self) -> Union[Block, None]:
        """Get the latest block in the blockchain."""
        blockchain = self.load_blockchain()
        return blockchain[-1] if blockchain else None
