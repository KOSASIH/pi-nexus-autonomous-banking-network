from typing import Optional

from components.pi_network import PiNetwork


class PiNetworkService:
    def __init__(self, config: dict):
        self.config = config
        self.pi_network = PiNetwork(self.create_api_client())

    def create_api_client(self) -> dict:
        # Implement Pi Network API client creation
        pass

    def get_transaction(self, hash: str) -> Optional[Transaction]:
        return self.pi_network.get_transaction(hash)

    def get_account_balance(self, address: str) -> Optional[float]:
        return self.pi_network.get_account_balance(address)

    def submit_transaction(
        self, sender: str, receiver: str, amount: float
    ) -> Optional[Transaction]:
        return self.pi_network.submit_transaction(sender, receiver, amount)
