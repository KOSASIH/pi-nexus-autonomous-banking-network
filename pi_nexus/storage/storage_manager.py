import json
import os

import web3


class StorageManager:
    def __init__(self, provider_url):
        self.provider_url = provider_url
        self.web3 = web3.Web3(web3.HTTPProvider(self.provider_url))
        self.chain_id = self.web3.eth.chain_id

    def deploy_storage_contract(self, storage_contract_path):
        with open(storage_contract_path) as f:
            storage_contract_code = f.read()

        storage_contract = self.web3.eth.contract(
            abi=storage_contract_code["abi"], bytecode=storage_contract_code["bin"]
        )
        tx_hash = storage_contract.constructor().transact(
            {"from": self.web3.eth.defaultAccount, "gas": 1000000}
        )
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)
        storage_address = tx_receipt["contractAddress"]

        return storage_address

    def call_storage_function(self, storage_address, function_name, *args):
        storage_contract = self.web3.eth.contract(
            address=storage_address, abi=self.get_storage_contract_abi()
        )
        result = storage_contract.functions[function_name](*args).call()

        return result

    def get_storage_contract_abi(self):
        # Implement a function to retrieve the ABI of the storage contract based on the chain ID
        pass

    def store_file(self, storage_address, name, hash):
        tx_hash = storage_contract.functions.storeFile(name, hash).transact(
            {"from": self.web3.eth.defaultAccount, "gas": 1000000}
        )
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)

        return tx_receipt

    def get_file(self, storage_address, owner, file_id):
        file = storage_contract.functions.getFile(owner, file_id).call()

        return file

    def delete_file(self, storage_address, owner, file_id):
        tx_hash = storage_contract.functions.deleteFile(owner, file_id).transact(
            {"from": self.web3.eth.defaultAccount, "gas": 1000000}
        )
        tx_receipt = self.web3.eth.waitForTransactionReceipt(tx_hash)

        return tx_receipt
