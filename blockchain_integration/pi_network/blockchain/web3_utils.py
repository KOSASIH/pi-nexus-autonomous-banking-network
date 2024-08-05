import web3
from web3 import Web3
from web3.middleware import geth_poa_middleware

class Web3Utils:
    def __init__(self, provider_url):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    def get_block_number(self):
        return self.w3.eth.blockNumber

    def get_transaction_count(self, address):
        return self.w3.eth.getTransactionCount(address)

    def get_balance(self, address):
        return self.w3.eth.get_balance(address)

    def send_transaction(self, from_address, to_address, value):
        tx = {
            'nonce': self.w3.eth.getTransactionCount(from_address),
            'gasPrice': self.w3.toWei('20', 'gwei'),
            'gas': 100000,
            'to': to_address,
            'value': self.w3.toWei(value, 'ether')
        }
        signed_tx = self.w3.eth.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return tx_hash.hex()

    def get_transaction_receipt(self, tx_hash):
        return self.w3.eth.getTransactionReceipt(tx_hash)

    def get_block(self, block_number):
        return self.w3.eth.getBlock(block_number)

    def get_transaction(self, tx_hash):
        return self.w3.eth.getTransaction(tx_hash)
