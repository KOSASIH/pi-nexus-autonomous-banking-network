# blockchain/miner.py
import logging

logger = logging.getLogger(__name__)


class Miner:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def mine_pending_transactions(self, mining_reward_address):
        try:
            # ...
            self.blockchain.add_block(new_block)
            logger.info("New block added to the blockchain")
        except Exception as e:
            logger.error(f"Error mining pending transactions: {e}")
