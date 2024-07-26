import hashlib
from typing import List, Dict

class BlockchainModel:
    def __init__(self, name: str):
        self.name = name
        self.blocks = []
        self.current_block_height = 0

    def add_block(self, block: Dict) -> None:
        # Add new block to blockchain
        self.blocks.append(block)
        self.current_block_height += 1

    def validate_block(self, block: Dict) -> bool:
        # Validate block data
        if not block or not block["transactions"]:
            return False

        # Calculate block hash
        block_hash = hashlib.sha256(f"{block['block_height']}{block['transactions']}".encode()).hexdigest()

        # Check if block hash matches block ID
        return block_hash == block["block_id"]

    def get_block(self, block_height: int) -> Dict:
        # Retrieve specific block from blockchain
        for block in self.blocks:
            if block["block_height"] == block_height:
                return block
        return None

    def get_transaction(self, transaction_id: str) -> Dict:
        # Retrieve specific transaction from blockchain
        for block in self.blocks:
            for transaction in block["transactions"]:
                if transaction["transaction_id"] == transaction_id:
                    return transaction
        return None
