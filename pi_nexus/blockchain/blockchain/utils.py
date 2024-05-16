# blockchain/utils.py
def calculate_hash(index, previous_hash, timestamp, data) -> str:
    """Calculate the hash of a block."""
    # ...

# blockchain/miner.py
from .utils import calculate_hash

class Miner:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def mine_pending_transactions(self, mining_reward_address):
        # ...
        new_block = Block(index, previous_hash, timestamp, data, calculate_hash(index, previous_hash, timestamp, data))
        # ...
