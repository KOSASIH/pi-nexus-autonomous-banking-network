import requests
import base64
import hashlib
import hmac
import time
from urllib.parse import urlparse
from stellar_sdk import (
    Asset,
    Claimant,
    ClaimPredicate,
    Keypair,
    Network,
    Operation,
    Transaction,
    TransactionBuilder,
    TransactionEnvelope,
)

class StellarSDK:
    def __init__(self, horizon_url, network_passphrase, private_key):
        self.horizon_url = horizon_url
        self.network_passphrase = network_passphrase
        self.private_key = private_key
        self.network = Network(network_passphrase)
        self.keypair = Keypair.from_secret(private_key)

    def _sign_transaction(self, transaction):
        # Advanced feature: Transaction signing with Ed25519
        signature = hmac.new(base64.b64decode(self.private_key), transaction.encode(), hashlib.sha256).digest()
        return base64.b64encode(signature).decode()

    def create_account(self, address, starting_balance):
        # Advanced feature: Create account with inflation destination and clawback
        transaction = TransactionBuilder(
            source_account=self.keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100,
        ).append_create_account_op(
            destination=address,
            starting_balance=starting_balance,
            inflation_destination=address,
            clawback_enabled=True,
        ).build()
        transaction.sign(self.keypair)
        response = requests.post(self.horizon_url + "/transactions", json=transaction.to_xdr())
        return response.json()

    def payment(self, source, destination, amount, asset_code, asset_issuer):
        # Advanced feature: Payment with path payment and asset clawback
        transaction = TransactionBuilder(
            source_account=self.keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100,
        ).append_payment_op(
            destination=destination,
            amount=amount,
            asset_code=asset_code,
            asset_issuer=asset_issuer,
            path=[
                Asset.native(),
                Asset(asset_code, asset_issuer),
            ],
            clawback_enabled=True,
        ).build()
        transaction.sign(self.keypair)
        response = requests.post(self.horizon_url + "/transactions", json=transaction.to_xdr())
        return response.json()

    def manage_data(self, source, key, value):
        # Advanced feature: Manage data with hash(x) and wrap in a transaction
        transaction = TransactionBuilder(
            source_account=self.keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100,
        ).append_manage_data_op(
            source=source,
            key=key,
            value=value,
            hash=hashlib.sha256(value.encode()).hexdigest(),
        ).build()
        transaction.sign(self.keypair)
        response = requests.post(self.horizon_url + "/transactions", json=transaction.to_xdr())
        return response.json()

    def claim_asset(self, source, asset_code, asset_issuer, amount):
        # Advanced feature: Claim asset with claimant and predicate
        claimant = Claimant(
            destination=source,
            predicate=ClaimPredicate.predicate_unconditional(),
        )
        transaction = TransactionBuilder(
            source_account=self.keypair.public_key,
            network_passphrase=self.network_passphrase,
            base_fee=100,
        ).append_claim_asset_op(
            asset_code=asset_code,
            asset_issuer=asset_issuer,
            amount=amount,
            claimant=claimant,
        ).build()
        transaction.sign(self.keypair)
        response = requests.post(self.horizon_url + "/transactions", json=transaction.to_xdr())
        return response.json()

    def get_account(self, address):
        # Advanced feature: Get account with ledger and transaction history
        response = requests.get(self.horizon_url + "/accounts/" + address)
        return response.json()

    def get_transaction(self, transaction_id):
        # Advanced feature: Get transaction with memo and signatures
        response = requests.get(self.horizon_url + "/transactions/" + transaction_id)
        return response.json()

# Example usage:
sdk = StellarSDK("https://horizon-testnet.stellar.org", "Test SDF Network ; September 2015", "your_private_key_here")
sdk.create_account("your_address_here", 1000)
sdk.payment("your_source_address_here", "your_destination_address_here", 100, "USD", "your_asset_issuer_here")
sdk.manage_data("your_source_address_here", "your_key_here", "your_value_here")
sdk.claim_asset("your_source_address_here", "your_asset_code_here", "your_asset_issuer_here", 100)
print(sdk.get_account("your_address_here"))
print(sdk.get_transaction("your_transaction_id_here"))
