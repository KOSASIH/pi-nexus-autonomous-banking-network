import os
import json
from web3 import Web3
from openzeppelin-sdk import OpenZeppelinSDK

class SmartContractInteractor:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(os.environ.get("BLOCKCHAIN_NODE_URL")))
        self.sdk = OpenZeppelinSDK(self.web3)

    def deploy_contract(self, contract_name, args):
        # Deploy contract using OpenZeppelin SDK
        contract = self.sdk.compile(contract_name)
        tx_hash = self.sdk.deploy(contract, args)
        return tx_hash

    def interact_with_contract(self, contract_address, function_name, args):
        # Interact with deployed contract using OpenZeppelin SDK
        contract = self.sdk.at(contract_address)
        result = contract.functions[function_name](*args).call()
        return result

# Example usage
interactor = SmartContractInteractor()
tx_hash = interactor.deploy_contract("MyContract", ["arg1", "arg2"])
print(tx_hash)
result = interactor.interact_with_contract("0x...", "myFunction", ["arg1", "arg2"])
print(result)
