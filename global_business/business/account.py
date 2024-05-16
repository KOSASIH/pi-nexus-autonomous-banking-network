from uuid import uuid4
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Account:
    account_number: str
    account_type: str
    balance: float
    transactions: list

    def __post_init__(self):
        self.account_number = str(uuid4())[:8]
        self.transactions = []

    def deposit(self, amount: float):
        self.balance += amount
        self.transactions.append({
            "type": "deposit",
            "amount": amount,
            "balance": self.balance,
            "timestamp": datetime.now()
        })

    def withdraw(self, amount: float):
        if self.balance < amount:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        self.transactions.append({
            "type": "withdrawal",
            "amount": amount,
            "balance": self.balance,
            "timestamp": datetime.now()
        })
