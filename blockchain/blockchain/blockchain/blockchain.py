from typing import List
from block import Block

class Blockchain:
    """
    A class representing the blockchain.

    Attributes:
        chain (List[Block]): The list of blocks in the blockchain.
    """

    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self) -> Block:
        """Create the first block in the blockchain."""
        return Block(index=0, previous_hash="0", data="Genesis Block")

    def add_block(self, data: Any):
        """Add a new block to the blockchain."""
        previous_block = self.chain[-1]
        new_block = Block(index=len(self.chain), previous_hash=previous_block.hash, data=data)
        self.chain.append(new_block)

    def is_valid(self) -> bool:
        """Check the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
return False

        return True
