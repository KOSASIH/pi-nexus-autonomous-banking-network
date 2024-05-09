import requests
from bitcoinrpc.authproxy import AuthServiceProxy

class Bitcoin:
    def __init__(self, node_url):
        self.node_url = node_url
        self.rpc_connection = AuthServiceProxy(node_url)

    def get_balance(self, address):
        balance = self.rpc_connection.getbalance(address, minconf=1)
        return balance

    def send_transaction(self, sender, receiver, amount):
        tx_hash = self.rpc_connection.sendtoaddress(receiver, amount)
        return tx_hash
