import hashlib
import logging
import random
import time
from typing import Dict, List, Tuple


class Transaction:
    def __init__(self, sender: str, receiver: str, amount: float):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = int(time.time())
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        return hashlib.sha256(str(self.__dict__).encode("utf-8")).hexdigest()


class Node:
    def __init__(self, node_id: str, public_key: str, private_key: str):
        self.node_id = node_id
        self.public_key = public_key
        self.private_key = private_key
        self.balance = 0
        self.transactions = []

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

    def calculate_balance(self) -> float:
        total = 0
        for transaction in self.transactions:
            if transaction.sender == self.node_id:
                total -= transaction.amount
            elif transaction.receiver == self.node_id:
                total += transaction.amount
        return total


class Block:
    def __init__(
        self,
        index: int,
        previous_hash: str,
        timestamp: int,
        transactions: List[Transaction],
        nonce: int,
        hash: str,
    ):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.hash = hash

    def calculate_hash(self) -> str:
        return hashlib.sha256(str(self.__dict__).encode("utf-8")).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4
        self.mining_reward = 10
        self.nodes = set()

    def create_genesis_block(self) -> Block:
        return Block(0, "0" * 64, int(time.time()), [], 0, "0" * 64)

    def add_block(self, block: Block):
        block.previous_hash = self.get_latest_block().hash
        block.hash = block.calculate_hash()
        self.chain.append(block)

    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

            if not self.is_valid_difficulty(current_block):
                return False

        return True

    def is_valid_difficulty(self, block: Block) -> bool:
        return block.hash[0 : self.difficulty] == "0" * self.difficulty

    def mine_block(self, node: Node):
        last_block = self.get_latest_block()
        new_block = Block(
            len(self.chain), last_block.hash, int(time.time()), node.transactions, 0, ""
        )

        while not self.is_valid_difficulty(new_block):
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()

        self.add_block(new_block)
        node.balance += self.mining_reward

    def add_node(self, node: Node):
        self.nodes.add(node)

    def replace_chain(self):
        longest_chain = None
        max_length = len(self.chain)

        for node in self.nodes:
            chain = node.chain
            length = len(chain)

            if length > max_length and self.is_valid(chain):
                max_length = length
                longest_chain = chain

        if longest_chain:
            self.chain = longest_chain

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def get_balance(self, node_id: str) -> float:
        for node in self.nodes:
            if node.node_id == node_id:
                return node.calculate_balance()

        return 0

    def get_transaction_history(self, node_id: str) -> List[Transaction]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node.transactions

        return []

    def get_block_by_index(self, index: int) -> Block:
        return self.chain[index]

    def get_block_by_hash(self, hash: str) -> Block:
        for block in self.chain:
            if block.hash == hash:
                return block

        return None
