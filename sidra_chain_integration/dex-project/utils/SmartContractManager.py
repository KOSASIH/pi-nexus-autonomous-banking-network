import web3
from web3.contract import Contract

class SmartContractManager:
    def __init__(self, blockchain_url, contract_address):
        self.w3 = web3.Web3(web3.providers.HttpProvider(blockchain_url))
        self.contract = Contract(self.w3, contract_address)

    def deploy_contract(self, contract_code):
        tx_hash = self.w3.eth.send_transaction({'from': self.w3.eth.accounts[0], 'gas': 200000, 'data': contract_code})
        return tx_hash

    def execute_contract_function(self, function_name, args):
        tx_hash = self.contract.functions[function_name](*args).transact({'from': self.w3.eth.accounts[0]})
        return tx_hash
