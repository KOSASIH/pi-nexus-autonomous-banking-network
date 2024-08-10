from models import Transaction

class TransactionService:
  def __init__(self):
    self.transactions = []

  def get_transactions(self):
    return self.transactions

  def add_transaction(self, transaction):
    self.transactions.append(transaction)
    return self.transactions

  def validate_transaction(self, transaction):
    if transaction.sender == transaction.recipient:
      return False
    if transaction.amount <= 0:
      return False
    return True
