# blockchain/wallet.py
from web3 import Web3


class Wallet:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.account_address = "0x..."

    def send_transaction(self, to_address, value):
        # implementation
        pass
