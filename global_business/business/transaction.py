from datetime import datetime


class Transaction:
    def __init__(self, account: "Account", transaction_type: str, amount: float):
        self.account = account
        self.transaction_type = transaction_type
        self.amount = amount
        self.timestamp = datetime.now()

    def process(self):
        if self.transaction_type == "deposit":
            self.account.deposit(self.amount)
        elif self.transaction_type == "withdrawal":
            self.account.withdraw(self.amount)
        else:
            raise ValueError("Invalid transaction type")
