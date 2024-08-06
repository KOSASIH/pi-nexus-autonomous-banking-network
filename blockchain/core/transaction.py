from typing import List

class Transaction:
    def __init__(self, sender: str, receiver: str, amount: float):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

    def __str__(self):
        return f"Transaction - Sender: {self.sender}, Receiver: {self.receiver}, Amount: {self.amount}"

class TransactionPool:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

    def get_transactions(self):
        return self.transactions
