from web3 import Web3
import json

class MultiSigWalletService:
    def __init__(self, contract_address, abi):
        self.w3 = Web3(Web3.HTTPProvider('https://your.ethereum.node'))
        self.contract = self.w3.eth.contract(address=contract_address, abi=abi)

    def submit_transaction(self, owner_address, to, value):
        tx_hash = self.contract.functions.submitTransaction(to, value).transact({'from': owner_address})
        return tx_hash.hex()

    def approve_transaction(self, owner_address, tx_index):
        tx_hash = self.contract.functions.approveTransaction(tx_index).transact({'from': owner_address})
        return tx_hash.hex()

    def execute_transaction(self, owner_address, tx_index):
        tx_hash = self.contract.functions.executeTransaction(tx_index).transact({'from': owner_address})
        return tx_hash.hex()

    def get_transaction(self, tx_index):
        return self.contract.functions.getTransaction(tx_index).call()

    def get_transaction_count(self):
        return self.contract.functions.getTransactionCount().call()
