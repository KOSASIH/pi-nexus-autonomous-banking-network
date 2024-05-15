# miner.py
import hashlib
import time
from blockchain import Blockchain
from wallet import Wallet

class Miner:
    def __init__(self, wallet: Wallet, blockchain: Blockchain):
        self.wallet = wallet
        self.blockchain = blockchain

    def mine_new_block(self, transactions: list) -> dict:
        """
        Mine a new block and add it to the blockchain.

        Args:
            transactions (list): List of transactions to include in the block.

        Returns:
            dict: The newly mined block.
        """
        # Create a new block with the given transactions
        block = {
            'index': self.blockchain.get_latest_block()['index'] + 1,
            'timestamp': int(time.time()),
            'transactions': transactions,
            'previous_hash': self.blockchain.get_latest_block()['hash'],
            'nonce': 0
        }

        # Mine the block by finding a hash that meets the difficulty requirement
        while not self._is_valid_hash(block):
            block['nonce'] += 1
            block['hash'] = self._calculate_hash(block)

        # Add the block to the blockchain
        self.blockchain.add_block(block)

        # Reward the miner with Pi coins
        self.wallet.add_coins(10)  # 10 Pi coins per block

        return block

    def _is_valid_hash(self, block: dict) -> bool:
        """
        Check if the block's hash meets the difficulty requirement.

        Args:
            block (dict): The block to check.

        Returns:
            bool: True if the hash is valid, False otherwise.
        """
        target = '0' * self.blockchain.difficulty
        return block['hash'][:self.blockchain.difficulty] == target

    def _calculate_hash(self, block: dict) -> str:
        """
        Calculate the hash of the block.

        Args:
            block (dict): The block to hash.

        Returns:
            str: The hash of the block.
        """
        block_string = str(block).encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()
