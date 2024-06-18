from stellar_sdk.client import StellarClient
from models.stellar_account import StellarAccount
from models.stellar_asset import StellarAsset
from models.stellar_transaction import StellarTransaction

class StellarService:
    def __init__(self, stellar_client, stellar_account, stellar_asset):
        self.stellar_client = stellar_client
        self.stellar_account = stellar_account
        self.stellar_asset = stellar_asset

    def create_transaction(self, destination_account, amount):
        transaction = StellarTransaction(self.stellar_account, destination_account, self.stellar_asset, amount)
        return transaction

    def submit_transaction(self, transaction):
        # Submit transaction to Stellar network
        pass
