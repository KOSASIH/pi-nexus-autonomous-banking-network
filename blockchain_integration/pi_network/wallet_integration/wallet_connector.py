import requests
import json
from stellar_sdk import Server, Keypair, TransactionBuilder, Operation

class WalletConnector:
    def __init__(self, pi_testnet_url, pi_mainnet_url, stellar_sdk_server):
        self.pi_testnet_url = pi_testnet_url
        self.pi_mainnet_url = pi_mainnet_url
        self.stellar_sdk_server = stellar_sdk_server

    def create_payment(self, recipient_address, amount, payment_identifier, user_uid):
        # Create a payment operation
        payment = Operation.payment(
            destination=recipient_address,
            asset=self.stellar_sdk_server.Asset.native(),
            amount=str(amount)
        )

        # Build the transaction
        transaction = TransactionBuilder(
            self.stellar_sdk_server,
            source_account=self.stellar_sdk_server.load_account(self.stellar_sdk_server.keypair.secret()),
            network_passphrase="Pi Testnet",
            timebounds=self.stellar_sdk_server.fetch_timebounds(180)
        ).add_operation(payment).add_memo(StellarSdk.Memo.text(payment_identifier)).build()

        # Sign the transaction
        transaction.sign(self.stellar_sdk_server.keypair)

        # Submit the transaction to the Pi blockchain
        response = requests.post(self.pi_testnet_url + "/v2/payments", json={
            "txid": transaction.to_xdr(),
            "payment_identifier": payment_identifier,
            "user_uid": user_uid
        })

        return response.json()

    def complete_payment(self, payment_identifier, txid):
        # Complete the payment by sending API request to `/complete` endpoint
        response = requests.post(self.pi_testnet_url + f"/v2/payments/{payment_identifier}/complete", json={"txid": txid})

        return response.json()
