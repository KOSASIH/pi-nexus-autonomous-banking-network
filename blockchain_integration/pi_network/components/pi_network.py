from typing import Optional


class Transaction:
    def __init__(
        self, hash: str, sender: str, receiver: str, amount: float, timestamp: datetime
    ):
        self.hash = hash
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp


class Account:
    def __init__(self, address: str, balance: float):
        self.address = address
        self.balance = balance


class PiNetwork:
    def __init__(self, api_client):
        self.api_client = api_client

    def get_transaction(self, hash: str) -> Optional[Transaction]:
        # Implement Pi Network API call to get transaction by hash
        pass

    def get_account_balance(self, address: str) -> Optional[float]:
        # Implement Pi Network API call to get account balance
        pass

    def submit_transaction(
        self, sender: str, receiver: str, amount: float
    ) -> Optional[Transaction]:
        # Implement Pi Network API call to submit transaction
        pass
