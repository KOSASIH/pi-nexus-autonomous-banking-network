# eonix_network.py
import hashlib
import json
import time
from eonix_block import EonixBlock
from eonix_transaction import EonixTransaction

class EonixNetwork:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = 2
        self.mining_reward = 10

    def add_block(self, block: EonixBlock):
        if self.is_valid_block(block):
            self.chain.append(block)
            self.pending_transactions = []
            return True
        return False

    def is_valid_block(self, block: EonixBlock):
        if not block.validate():
            return False
        if len(self.chain) > 0 and block.get_previous_block_hash() != self.chain[-1].get_block_hash():
            return False
        if not self.is_valid_proof(block):
            return False
        return True

    def is_valid_proof(self, block: EonixBlock):
        proof = block.get_block_hash()
        return proof.startswith("0" * self.difficulty)

    def mine_block(self):
        if not self.pending_transactions:
            return None
        new_block = EonixBlock(len(self.chain), self.chain[-1].get_block_hash() if self.chain else "0x0000000000000000000000000000000000000000000000000000000000000000", self.pending_transactions, int(time.time()))
        new_block_hash = new_block.get_block_hash()
        while not new_block_hash.startswith("0" * self.difficulty):
            new_block.increment_nonce()
            new_block_hash = new_block.get_block_hash()
        self.add_block(new_block)
        return new_block

    def add_transaction(self, transaction: EonixTransaction):
        if transaction.validate():
            self.pending_transactions.append(transaction)
            return True
        return False

    def get_chain(self):
        return self.chain

    def get_pending_transactions(self):
        return self.pending_transactions

    def to_dict(self):
        chain_dict = [block.to_dict() for block in self.chain]
        pending_transactions_dict = [tx.to_dict() for tx in self.pending_transactions]
        return {
            "chain": chain_dict,
            "pending_transactions": pending_transactions_dict,
            "difficulty": self.difficulty,
            "mining_reward": self.mining_reward
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, network_dict):
        chain = [EonixBlock.from_dict(block_dict) for block_dict in network_dict["chain"]]
        pending_transactions = [EonixTransaction.from_dict(tx_dict) for tx_dict in network_dict["pending_transactions"]]
        network = cls()
        network.chain = chain
        network.pending_transactions = pending_transactions
        network.difficulty = network_dict["difficulty"]
        network.mining_reward = network_dict["mining_reward"]
        return network

    @classmethod
    def from_json(cls, network_json):
        network_dict = json.loads(network_json)
        return cls.from_dict(network_dict)
