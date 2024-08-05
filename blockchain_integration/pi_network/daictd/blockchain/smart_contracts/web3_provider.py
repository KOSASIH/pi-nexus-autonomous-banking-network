import os
from web3 import Web3, HTTPProvider

class Web3Provider:
    def __init__(self, project_id):
        self.project_id = project_id
        self.w3 = Web3(HTTPProvider(f'https://mainnet.infura.io/v3/{project_id}'))

    def get_web3(self):
        return self.w3

    def get_contract(self, contract_address, abi):
        return self.w3.eth.contract(address=contract_address, abi=abi)

    def deploy_contract(self, contract_interface):
        tx_hash = self.w3.eth.contract(bytecode=contract_interface['bytecode'], abi=contract_interface['abi']).constructor().transact()
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt.contractAddress
