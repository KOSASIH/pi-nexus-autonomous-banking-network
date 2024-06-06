import hashlib
import json
from web3 import Web3, HTTPProvider

class BlockchainInterface:
    def __init__(self, network_type, network_url):
        self.network_type = network_type
        self.network_url = network_url
        self.web3 = Web3(HTTPProvider(network_url))

    def get_block_number(self):
        return self.web3.eth.block_number

    def get_transaction_count(self, address):
        return self.web3.eth.get_transaction_count(address)

    def send_transaction(self, from_address, to_address, value, gas, gas_price):
        tx = {
            'from': from_address,
            'to': to_address,
            'value': value,
            'gas': gas,
            'gasPrice': gas_price
        }
        signed_tx = self.web3.eth.account.sign_transaction(tx)
        self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)

    def deploy_smart_contract(self, contract_code, from_address, gas, gas_price):
        tx = {
            'from': from_address,
            'data': contract_code,
            'gas': gas,
            'gasPrice': gas_price
        }
        signed_tx = self.web3.eth.account.sign_transaction(tx)
        self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)

    def call_smart_contract(self, contract_address, function_name, args, from_address, gas, gas_price):
        contract = self.web3.eth.contract(address=contract_address, abi=json.loads(open('contract.abi', 'r').read()))
        tx = contract.functions[function_name](*args).buildTransaction({
            'from': from_address,
            'gas': gas,
            'gasPrice': gas_price
        })
        signed_tx = self.web3.eth.account.sign_transaction(tx)
        self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)

    def get_balance(self, address):
        return self.web3.eth.get_balance(address)

    def get_transaction_receipt(self, tx_hash):
        return self.web3.eth.get_transaction_receipt(tx_hash)
