# services/transaction_service.py
from inject import inject
from models.transaction import Transaction

class TransactionService:
    @inject
    def __init__(self, transaction_repository):
        self.transaction_repository = transaction_repository

    def create_transaction(self, sender: str, receiver: str, amount: int):
        transaction = Transaction(sender=sender, receiver=receiver, amount=amount)
        self.transaction_repository.save(transaction)
        return transaction

    def get_transactions(self, username: str):
        return self.transaction_repository.get_by_username(username)
