import hashlib
import json
from time import time
from typing import Any


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2
        self.pending_transactions = []

    def create_genesis_block(self) -> Block:
        return Block(
            index=0, previous_hash="0" * 64, timestamp=time(), transactions=[], nonce=0
        )

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, block: Block):
        block.previous_hash = self.get_latest_block().hash
        block.mine_block(self.difficulty)
        self.chain.append(block)

    def add_transaction(self, transaction: Transaction):
        self.pending_transactions.append(transaction)

    def create_merkle_tree(self):
        merkle_tree = []
        for transaction in self.pending_transactions:
            merkle_tree.append(transaction.to_dict())
        while len(merkle_tree) > 1:
            parent_nodes = []
            for i in range(0, len(merkle_tree), 2):
                if i + 1 < len(merkle_tree):
                    left_node = merkle_tree[i]
                    right_node = merkle_tree[i + 1]
                    parent_nodes.append(self.hash_pair(left_node, right_node))
                else:
                    parent_nodes.append(merkle_tree[i])
            merkle_tree = parent_nodes
        return merkle_tree[0]

    def hash_pair(self, left_node: Any, right_node: Any) -> str:
        left_node_json = json.dumps(left_node, sort_keys=True).encode()
        right_node_json = json.dumps(right_node, sort_keys=True).encode()
        pair_data = left_node_json + right_node_json
        return hashlib.sha256(pair_data).hexdigest()

    def validate_transaction(self, transaction: Transaction) -> bool:
        sender_balance = 0
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == transaction.sender:
                    sender_balance += tx.amount
                if tx.receiver == transaction.sender:
                    sender_balance -= tx.amount
        return sender_balance >= transaction.amount

    def validate_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True
