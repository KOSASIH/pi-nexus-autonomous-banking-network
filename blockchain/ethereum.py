import requests
from web3 import Web3

class Ethereum:
    def __init__(self, node_url):
        self.node_url = node_url
        self.web3 = Web3(Web3.HTTPProvider(self.node_url))

    def get_balance(self, address):
        balance = self.web3.eth.get_balance(address)
        return self.web3.fromWei(balance, "ether")

    def send_transaction(self, sender, receiver, amount):
        tx_hash = self.web3.eth.send_transaction({
            "from": sender,
            "to": receiver,
            "value": self.web3.toWei(amount, "ether")
        })
        return tx_hash
