from typing import List

class Wallet:
    def __init__(self, public_key: str, private_key: str):
        self.public_key = public_key
        self.private_key = private_key

    def create_transaction(self, recipient: str, amount: float):
        transaction = Transaction(self.public_key, recipient, amount)
        return transaction

class WalletPool:
    def __init__(self):
        self.wallets = []

    def add_wallet(self, wallet: Wallet):
        self.wallets.append(wallet)

    def get_wallet(self, public_key: str):
        for wallet in self.wallets:
            if wallet.public_key == public_key:
                return wallet
        return None
