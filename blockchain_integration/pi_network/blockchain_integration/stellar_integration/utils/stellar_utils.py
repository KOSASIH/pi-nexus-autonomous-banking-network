import hashlib
import base64
from stellar_sdk import Server, TransactionBuilder, Asset, Keypair

def generate_keypair():
    return Keypair.random()

def get_account_info(account_id):
    server = Server(horizon_url="https://horizon.stellar.org")
    account = server.accounts().account_id(account_id).call()
    return account

def create_transaction(source, destination, asset, amount):
    transaction = TransactionBuilder(
        source_account=source,
        network_passphrase=network_passphrase,
        base_fee=fee
    ).append_payment_op(
        destination=destination,
        asset_code=asset,
        amount=str(amount)
    ).build()
    return transaction

def sign_transaction(transaction, keypair):
    transaction.sign(keypair)
    return transaction

def submit_transaction(transaction):
    server = Server(horizon_url="https://horizon.stellar.org")
    response = server.submit_transaction(transaction)
    return response
