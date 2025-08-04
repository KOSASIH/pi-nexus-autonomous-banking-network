from datetime import datetime

from models.account import Account


class Transaction:
    def __init__(
        self,
        account_from: Account,
        account_to: Account,
        amount: float,
        timestamp: datetime = None,
    ):
        self.account_from = account_from
        self.account_to = account_to
        self.amount = amount
        self.timestamp = timestamp or datetime.now()

    def debit(self):
        self.account_from.balance -= self.amount
        self.account_to.balance += self.amount

    def credit(self):
        self.account_from.balance += self.amount
        self.account_to.balance -= self.amount

    def process(self):
        if self.amount > 0:
            self.debit()
        else:
            self.credit()


class TransferTransaction(Transaction):
    def __init__(
        self,
        account_from: Account,
        account_to: Account,
        amount: float,
        timestamp: datetime = None,
    ):
        super().__init__(account_from, account_to, amount, timestamp)

    def process(self):
        if self.amount > 0:
            if self.account_from.balance >= self.amount:
                super().debit()
            else:
                raise InsufficientFundsError(
                    f"Insufficient funds in account {self.account_from.number}"
                )
        else:
            super().credit()


class DepositTransaction(Transaction):
    def __init__(self, account: Account, amount: float, timestamp: datetime = None):
        super().__init__(None, account, amount, timestamp)

    def process(self):
        super().credit()


class WithdrawalTransaction(Transaction):
    def __init__(self, account: Account, amount: float, timestamp: datetime = None):
        super().__init__(account, None, -amount, timestamp)

    def process(self):
        super().debit()


class InsufficientFundsError(Exception):
    def __init__(self, message):
        super().__init__(message)
