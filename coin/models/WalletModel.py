import hashlib
from typing import List, Dict

class WalletModel:
    def __init__(self, address: str):
        self.address = address
        self.balance = 0
        self.transactions = []

    def generate_address(self) -> str:
        # Generate new wallet address
        return hashlib.sha256(f"{self.address}{self.balance}".encode()).hexdigest()

    def send_coins(self, amount: float, receiver_address: str) -> None:
        # Send coins to another wallet
        self.balance -= amount
        self.transactions.append({"amount": amount, "receiver_address": receiver_address})

    def receive_coins(self, amount: float, sender_address: str) -> None:
        # Receive coins from another wallet
        self.balance += amount
        self.transactions.append({"amount": amount, "sender_address": sender_address})

    def update_balance(self) -> None:
        # Update wallet balance
        self.balance = sum(transaction["amount"] for transaction in self.transactions)

    def to_dict(self) -> Dict:
        # Convert wallet data to dictionary
        return {
            "address": self.address,
            "balance": self.balance,
            "transactions": self.transactions
        }
