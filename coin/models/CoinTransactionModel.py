import hashlib
import time
from typing import Dict

class CoinTransactionModel:
    def __init__(self, transaction_id: str, sender_address: str, receiver_address: str, amount: float, timestamp: int = None):
        self.transaction_id = transaction_id
        self.sender_address = sender_address
        self.receiver_address = receiver_address
        self.amount = amount
        self.timestamp = timestamp if timestamp else int(time.time())

    def validate(self) -> bool:
        # Validate transaction data
        if not self.sender_address or not self.receiver_address or self.amount <= 0:
            return False

        # Calculate transaction hash
        transaction_hash = hashlib.sha256(f"{self.sender_address}{self.receiver_address}{self.amount}{self.timestamp}".encode()).hexdigest()

        # Check if transaction hash matches transaction ID
        return transaction_hash == self.transaction_id

    def calculate_fee(self) -> float:
        # Calculate transaction fee based on amount and sender address
        fee = self.amount * 0.01
        if self.sender_address.startswith(" premium"):
            fee *= 0.5
        return fee

    def update_status(self, status: str) -> None:
        # Update transaction status
        self.status = status

    def to_dict(self) -> Dict:
        # Convert transaction data to dictionary
        return {
            "transaction_id": self.transaction_id,
            "sender_address": self.sender_address,
            "receiver_address": self.receiver_address,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "status": self.status
        }
