import json
from stellar_sdk import Keypair

class StellarWallet:
    def __init__(self, secret_key, public_key):
        self.secret_key = secret_key
        self.public_key = public_key
        self.keypair = Keypair.from_secret(secret_key)

    def get_account(self):
        return self.keypair.public_key

    def sign_transaction(self, transaction):
        return self.keypair.sign(transaction)

    def get_secret_key(self):
        return self.secret_key

    def get_public_key(self):
        return self.public_key
