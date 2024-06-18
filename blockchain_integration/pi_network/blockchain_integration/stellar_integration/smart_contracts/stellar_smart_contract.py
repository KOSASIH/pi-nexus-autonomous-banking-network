from stellar_sdk import Server, Network, Keypair, TransactionBuilder
from stellar_utils import get_stellar_client, get_account_sequence
from stellar_transaction_builder import build_payment_transaction

class StellarSmartContract:
    def __init__(self, contract_seed: str, network: str):
        self.contract_keypair = Keypair.from_seed(contract_seed)
        self.contract_account = self.contract_keypair.account_id
        self.network = network
        self.stellar_client = get_stellar_client(network)

    def execute_payment(self, source_seed: str, destination_id: str, amount: int, asset_code: str):
        """Execute a payment operation using the smart contract"""
        source_keypair = Keypair.from_seed(source_seed)
        source_account = source_keypair.account_id
        sequence = get_account_sequence(source_account, self.stellar_client)
        transaction = build_payment_transaction(
            source_seed, destination_id, amount, asset_code, self.network
        )
        transaction.sign(self.contract_keypair)
        response = self.stellar_client.submit_transaction(transaction)
        return response

    def execute_create_account(self, source_seed: str, destination_id: str, starting_balance: int):
        """Execute a create account operation using the smart contract"""
        source_keypair = Keypair.from_seed(source_seed)
        source_account = source_keypair.account_id
        sequence = get_account_sequence(source_account, self.stellar_client)
        transaction = build_create_account_transaction(
            source_seed, destination_id, starting_balance, self.network
        )
        transaction.sign(self.contract_keypair)
        response = self.stellar_client.submit_transaction(transaction)
        return response
