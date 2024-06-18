from stellar_utils import get_stellar_client, get_account_sequence
from stellar_sdk import Account, TransactionBuilder, Network

class StellarAccountManager:
    def __init__(self, stellar_network: str):
        self.stellar_network = stellar_network
        self.stellar_client = get_stellar_client(stellar_network)

    def create_account(self, user_id: str) -> str:
        """Create a new Stellar account for the specified user"""
        account_id = f"{user_id}-stellar"
        if self.stellar_client.account_exists(account_id):
            return account_id
        keypair = self.stellar_client.keypair()
        account = Account(account_id, keypair.secret)
        self.stellar_client.create_account(account)
        return account_id

    def get_account_balance(self, user_id: str) -> int:
        """Return the balance of the specified user's Stellar account"""
        account_id = f"{user_id}-stellar"
        account = self.stellar_client.account(account_id)
        return account.balance

    def fund_account(self, user_id: str, amount: int) -> TransactionBuilder:
        """Fund the specified user's Stellar account with the specified amount"""
        account_id = f"{user_id}-stellar"
        sequence = get_account_sequence(account_id, self.stellar_client)
        transaction = TransactionBuilder(
            source_account=account_id,
            sequence=sequence,
            operations=[
                stellar_sdk.Operation.payment(
                    destination=account_id,
                    asset_code="XLM",
                    amount=str(amount)
                )
            ],
            network_passphrase=Network[self.stellar_network].network_passphrase
        )
        return transaction
