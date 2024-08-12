import os
import json
from web3 import Web3, HTTPProvider
from web3.contract import Contract

class DataSharingContract:
    def __init__(self, contract_address, abi, provider_url):
        self.contract_address = contract_address
        self.abi = abi
        self.provider_url = provider_url
        self.w3 = Web3(HTTPProvider(self.provider_url))
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)

    def register_data_provider(self, provider_address, data_hash):
        tx_hash = self.contract.functions.registerDataProvider(provider_address, data_hash).transact({'from': provider_address})
        self.w3.eth.waitForTransactionReceipt(tx_hash)

    def register_data_consumer(self, consumer_address):
        tx_hash = self.contract.functions.registerDataConsumer(consumer_address).transact({'from': consumer_address})
        self.w3.eth.waitForTransactionReceipt(tx_hash)

    def request_data(self, consumer_address, data_hash):
        tx_hash = self.contract.functions.requestData(consumer_address, data_hash).transact({'from': consumer_address})
        self.w3.eth.waitForTransactionReceipt(tx_hash)

    def provide_data(self, provider_address, data_hash, data):
        tx_hash = self.contract.functions.provideData(provider_address, data_hash, data).transact({'from': provider_address})
        self.w3.eth.waitForTransactionReceipt(tx_hash)

    def get_data(self, data_hash):
        data = self.contract.functions.getData(data_hash).call()
        return data

    def get_data_providers(self):
        providers = self.contract.functions.getDataProviders().call()
        return providers

    def get_data_consumers(self):
        consumers = self.contract.functions.getDataConsumers().call()
        return consumers

if __name__ == '__main__':
    contract_address = '0x...contract address...'
    abi = json.load(open('data_sharing_contract.abi', 'r'))
    provider_url = 'https://mainnet.infura.io/v3/...project id...'
    contract = DataSharingContract(contract_address, abi, provider_url)

    # Example usage:
    provider_address = '0x...provider address...'
    data_hash = '0x...data hash...'
    contract.register_data_provider(provider_address, data_hash)

    consumer_address = '0x...consumer address...'
    contract.register_data_consumer(consumer_address)

    contract.request_data(consumer_address, data_hash)
    data = contract.get_data(data_hash)
    print(data)
