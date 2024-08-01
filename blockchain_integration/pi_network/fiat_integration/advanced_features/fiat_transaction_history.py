# fiat_transaction_history.py

class FiatTransactionHistory:
    def __init__(self):
        self.transaction_history = []

    def add_transaction(self, transaction):
        self.transaction_history.append(transaction)

    def get_transaction_history(self):
        return self.transaction_history
