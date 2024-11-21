# services/wallet_service.py

from typing import Optional

from components.wallet import Wallet

from config import Config


class WalletService:
    def __init__(self, config: Config):
        self.config = config
        self.wallet = Wallet(self.create_api_client())

    def create_api_client(self) -> dict:
        # Implement wallet API client creation
        pass

    def get_balance(self, address: str) -> Optional[float]:
        return self.wallet.get_balance(address)

    def send_transaction(
        self, sender: str, receiver: str, amount: float
    ) -> Optional[Transaction]:
        return self.wallet.send_transaction(sender, receiver, amount)

    def get_transaction_history(self, address: str) -> List[Transaction]:
        return self.wallet.get_transaction_history(address)

    def generate_new_address(self) -> str:
        return self.wallet.generate_new_address()

    def import_private_key(self, private_key: str) -> bool:
        return self.wallet.import_private_key(private_key)

    def export_private_key(self, address: str) -> Optional[str]:
        return self.wallet.export_private_key(address)
