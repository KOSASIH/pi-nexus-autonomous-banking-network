from block_model import Block
from block_repository import BlockRepository
from utils import proof_of_work, calculate_difficulty

class Blockchain:
    """
    A class representing the blockchain.

    Attributes:
        repository (BlockRepository): The block repository for storing the blockchain.
        min_difficulty (int): The minimum difficulty for Proof-of-Work.
        time_between_blocks (int): The target time between blocks.
    """

    def __init__(self, min_difficulty: int = 1, time_between_blocks: int = 10):
        self.repository = BlockRepository()
        self.min_difficulty = min_difficulty
        self.time_between_blocks = time_between_blocks

    def create_genesis_block(self):
        """Create the first block in the blockchain."""
        block = Block(index=0, previous_hash="0", data="Genesis Block")
        self.repository.add_block(block)

    def add_block(self, data: Any):
        """Add a new block to the blockchain."""
        latest_block = self.repository.get_latest_block()
        if latest_block is None:
            self.create_genesis_block()
            latest_block = self.repository.get_latest_block()

        difficulty = calculate_difficulty(latest_block, self.min_difficulty, self.time_between_blocks)
        nonce = proof_of_work(latest_block.hash, data, difficulty)
        new_block = Block(index=latest_block.index + 1, previous_hash=latest_block.hash, data=data, nonce=nonce)
        self.repository.add_block(new_block)

    def get_blockchain(self) -> list[Block]:
        """Get the blockchain."""
        return self.repository.load_blockchain()
