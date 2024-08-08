import hashlib
from web3 import Web3

def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

def send_transaction(data, blockchain_url):
    w3 = Web3(Web3.HTTPProvider(blockchain_url))
    tx_hash = w3.eth.sendTransaction({'from': '0x...', 'to': '0x...', 'value': 1, 'data': data})
    return tx_hash
