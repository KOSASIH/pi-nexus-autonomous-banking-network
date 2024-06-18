from stellar_sdk import TransactionBuilder, Network
from stellar_utils import get_keypair_from_seed

def build_payment_transaction(source_seed: str, destination_id: str, amount: int, asset_code: str, network: str):
    """Return a Stellar transaction instance for a payment operation"""
    source_keypair = get_keypair_from_seed(source_seed)
    source_account = source_keypair.account_id
    sequence = get_account_sequence(source_account, get_stellar_client(network))
    transaction = TransactionBuilder(
        source_account=source_account,
        sequence=sequence,
        operations=[
            TransactionBuilder.payment_operation(
                destination=destination_id,
                amount=str(amount),
                asset_code=asset_code
            )
        ],
        network_passphrase=Network[network].network_passphrase
    )
    return transaction

def build_create_account_transaction(source_seed: str, destination_id: str, starting_balance: int, network: str):
    """Return a Stellar transaction instance for a create account operation"""
    source_keypair = get_keypair_from_seed(source_seed)
    source_account = source_keypair.account_id
    sequence = get_account_sequence(source_account, get_stellar_client(network))
    transaction = TransactionBuilder(
        source_account=source_account,
        sequence=sequence,
        operations=[
            TransactionBuilder.create_account_operation(
                destination=destination_id,
                starting_balance=str(starting_balance)
            )
        ],
        network_passphrase=Network[network].network_passphrase
    )
    return transaction
