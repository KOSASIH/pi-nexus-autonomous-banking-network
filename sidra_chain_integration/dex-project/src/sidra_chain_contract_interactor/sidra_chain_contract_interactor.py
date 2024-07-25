# sidra_chain_contract_interactor.py
from web3 import Web3

class SidraChainContractInteractor:
    def __init__(self, web3_provider, contract_address, abi):
        self.web3_provider = web3_provider
        self.contract_address = contract_address
        self.abi = abi
        self.web3 = Web3(Web3.HTTPProvider(self.web3_provider))

    def interact_with_contract(self, function_name, *args):
        # Interact with the Sidra Chain contract
        contract = self.web3.eth.contract(address=self.contract_address, abi=self.abi)
        function = getattr(contract.functions, function_name)
        tx_hash = function(*args).transact()
        return tx_hash
