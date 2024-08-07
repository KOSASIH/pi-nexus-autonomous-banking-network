import web3
from web3 import Web3
from web3.contract import Contract
from web3.providers import AutoProvider

class Web3Utils:
    def __init__(self, provider: str, contract_address: str, contract_abi: list):
        self.web3 = Web3(AutoProvider(provider))
        self.contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)

    def get_block_number(self) -> int:
        return self.web3.eth.block_number

    def get_transaction_count(self, address: str) -> int:
        return self.web3.eth.get_transaction_count(address)

    def send_transaction(self, from_address: str, to_address: str, value: int) -> str:
        tx_hash = self.web3.eth.send_transaction({
            'from': from_address,
            'to': to_address,
            'value': value
        })
        return tx_hash.hex()

    def call_contract_function(self, function_name: str, *args) -> any:
        return getattr(self.contract.functions, function_name)(*args).call()

    def transact_contract_function(self, function_name: str, *args) -> str:
        tx_hash = getattr(self.contract.functions, function_name)(*args).transact()
        return tx_hash.hex()
