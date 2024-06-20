import hashlib
from stellar_sdk import Server, Keypair, TransactionBuilder, Network

class StellarWalletManager:
    def __init__(self, network_passphrase, horizon_url):
        self.network_passphrase = network_passphrase
        self.horizon_url = horizon_url
        self.server = Server(horizon_url)

    def generate_keypair(self, seed_phrase):
        keypair = Keypair.from_secret(seed_phrase)
        return keypair

    def create_account(self, keypair, starting_balance):
        transaction = TransactionBuilder(
            source_account=keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100
        ).append_create_account_op(
            destination=keypair.public_key,
            starting_balance=starting_balance
        ).build()
        self.server.submit_transaction(transaction)

    def get_account_balance(self, public_key):
        account = self.server.accounts().account_id(public_key).call()
        return account.balances[0].balance

    def send_payment(self, source_keypair, destination_public_key, amount):
        transaction = TransactionBuilder(
            source_account=source_keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100
        ).append_payment_op(
            destination=destination_public_key,
            amount=amount,
            asset_code="XLM"
        ).build()
        self.server.submit_transaction(transaction)

    def get_transaction_history(self, public_key):
        transactions = self.server.transactions().for_account(public_key).call()
        return transactions
