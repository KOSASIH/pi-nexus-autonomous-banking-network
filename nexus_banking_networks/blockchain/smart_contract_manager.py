import json
from web3 import Web3, HTTPProvider

class SmartContractManager:
    def __init__(self, blockchain_interface):
        self.blockchain_interface = blockchain_interface

    def deploy_contract(self, contract_code, from_address, gas, gas_price):
        tx_hash = self.blockchain_interface.deploy_smart_contract(contract_code, from_address, gas, gas_price)
        return tx_hash

    def call_contract_function(self, contract_address, function_name, args, from_address, gas, gas_price):
        tx_hash = self.blockchain_interface.call_smart_contract(contract_address, function_name, args, from_address, gas, gas_price)
        return tx_hash

    def get_contract_balance(self, contract_address):
        return self.blockchain_interface.get_balance(contract_address)

    def get_contract_storage(self, contract_address):
        contract = self.blockchain_interface.web3.eth.contract(address=contract_address, abi=json.loads(open('contract.abi', 'r').read()))
        storage = contract.functions.storage().call()
        return storage
