# eonix_block.py
import hashlib
import json
from eonix_transaction import EonixTransaction

class EonixBlock:
    def __init__(self, block_number, previous_block_hash, transactions: list[EonixTransaction], timestamp):
        self.block_number = block_number
        self.previous_block_hash = previous_block_hash
        self.transactions = transactions
        self.timestamp = timestamp
        self.block_hash = self.generate_block_hash()

    def generate_block_hash(self):
        block_hash_input = f"{self.block_number}{self.previous_block_hash}{self.timestamp}{self.get_transactions_hash()}"
        block_hash = hashlib.sha256(block_hash_input.encode()).hexdigest()
        return block_hash

    def get_transactions_hash(self):
        transactions_hash_input = "".join([tx.get_transaction_id() for tx in self.transactions])
        transactions_hash = hashlib.sha256(transactions_hash_input.encode()).hexdigest()
        return transactions_hash

    def get_block_number(self):
        return self.block_number

    def get_previous_block_hash(self):
        return self.previous_block_hash

    def get_transactions(self):
        return self.transactions

    def get_timestamp(self):
        return self.timestamp

    def get_block_hash(self):
        return self.block_hash

    def to_dict(self):
        transactions_dict = [tx.to_dict() for tx in self.transactions]
        return {
            "block_number": self.block_number,
            "previous_block_hash": self.previous_block_hash,
            "transactions": transactions_dict,
            "timestamp": self.timestamp,
            "block_hash": self.block_hash
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, block_dict):
        transactions_dict = block_dict["transactions"]
        transactions = [EonixTransaction.from_dict(tx_dict) for tx_dict in transactions_dict]
        return cls(
            block_dict["block_number"],
            block_dict["previous_block_hash"],
            transactions,
            block_dict["timestamp"]
        )

    @classmethod
    def from_json(cls, block_json):
        block_dict = json.loads(block_json)
        return cls.from_dict(block_dict)

    def validate(self):
        # Implement block validation logic here
        # For example:
        if self.block_number <= 0:
            raise ValueError("Block number must be greater than 0")
        if not self.previous_block_hash:
            raise ValueError("Previous block hash is required")
        if not self.transactions:
            raise ValueError("Block must contain at least one transaction")
        # Add more validation rules as needed
        return True
