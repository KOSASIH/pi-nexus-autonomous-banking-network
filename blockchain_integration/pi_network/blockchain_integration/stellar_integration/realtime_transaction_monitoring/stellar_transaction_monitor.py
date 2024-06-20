import stellar_sdk
from stellar_sdk.horizon import Horizon

class StellarTransactionMonitor:
    def __init__(self, horizon_url: str, stellar_network: str):
        self.horizon_url = horizon_url
        self.stellar_network = stellar_network
        self.horizon = Horizon(horizon_url)

    def stream_transactions(self, wallet_address: str) -> None:
        for transaction in self.horizon.transactions().for_account(wallet_address).stream():
            # Process transaction here
            print(f"Received transaction: {transaction.hash}")

    def get_transaction_history(self, wallet_address: str, limit: int = 10) -> list:
        transactions = self.horizon.transactions().for_account(wallet_address).limit(limit).call()
        return transactions
