from typing import Optional


class Wallet:
    def __init__(self, api_client):
        self.api_client = api_client

    def get_balance(self, address: str) -> Optional[float]:
        # Implement Pi Network wallet API call to get balance
        pass

    def send_transaction(
        self, sender: str, receiver: str, amount: float
    ) -> Optional[Transaction]:
        # Implement Pi Network wallet API call to send transaction
        pass
