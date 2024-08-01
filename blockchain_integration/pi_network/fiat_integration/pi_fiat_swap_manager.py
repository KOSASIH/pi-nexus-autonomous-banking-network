import requests
from stellar_sdk import Server, TransactionBuilder, Asset, Memo

class PiFiatSwapManager:
    def __init__(self, pi_testnet, my_public_key, my_secret_seed):
        self.pi_testnet = pi_testnet
        self.my_public_key = my_public_key
        self.my_secret_seed = my_secret_seed

    def load_account(self):
        response = self.pi_testnet.loadAccount(self.my_public_key)
        return response

    def build_transaction(self, recipient_address, amount):
        payment = TransactionBuilder(
            self.load_account(),
            fee=self.pi_testnet.fetchBaseFee(),
            network_passphrase="Pi Testnet"
        ).addOperation(
            payment_op=PaymentOp(
                destination=recipient_address,
                asset=Asset.native(),
                amount=amount
            )
        ).addMemo(
            Memo.text("Payment ID")
        ).build()
        return payment

    def sign_transaction(self, transaction):
        keypair = Keypair.fromSecret(self.my_secret_seed)
        transaction.sign(keypair)
        return transaction

    def submit_transaction(self, transaction):
        response = self.pi_testnet.submitTransaction(transaction)
        return response
