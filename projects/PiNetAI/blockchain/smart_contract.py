import os
import json
from typing import Dict, List
from web3 import Web3

class SmartContract:
    def __init__(self, contract_address: str, contract_abi: List[Dict[str, str]]):
        self.contract_address = contract_address
        self.contract_abi = contract_abi
        self.w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))

    def get_contract(self):
        return self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)

    def call_function(self, function_name: str, args: List[str]) -> str:
        contract = self.get_contract()
        function = contract.functions[function_name]
        return function(*args).call()

    def send_transaction(self, function_name: str, args: List[str], from_address: str, private_key: str) -> str:
        contract = self.get_contract()
        function = contract.functions[function_name]
        tx = function(*args).buildTransaction({
            "gas": 2000000,
            "gasPrice": self.w3.toWei("20", "gwei"),
            "from": from_address,
            "nonce": self.w3.eth.getTransactionCount(from_address)
        })
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
        return self.w3.eth.sendRawTransaction(signed_tx.rawTransaction).hex()

    def get_event(self, event_name: str, from_block: int, to_block: int) -> List[Dict[str, str]]:
        contract = self.get_contract()
        event = contract.events[event_name]
        return event.getLogs(fromBlock=from_block, toBlock=to_block)

    def get_balance(self, address: str) -> int:
        return self.w3.eth.get_balance(address)

# Example usage:
contract_address = "0x...ContractAddress..."
contract_abi = json.loads("[{\"inputs\":[],\"name\":\"exampleFunction\",\"outputs\":[{\"internalType\":\"uint256\",\"name\":\"\",\"type\":\"uint256\"}],\"stateMutability\":\"view\",\"type\":\"function\"}]")
smart_contract = SmartContract(contract_address, contract_abi)

# Call a function
result = smart_contract.call_function("exampleFunction", ["arg1", "arg2"])
print(result)

# Send a transaction
tx_hash = smart_contract.send_transaction("exampleFunction", ["arg1", "arg2"], "0x...FromAddress...", "0x...PrivateKey...")
print(tx_hash)

# Get an event
events = smart_contract.get_event("exampleEvent", 100, 200)
print(events)

# Get balance
balance = smart_contract.get_balance("0x...Address...")
print(balance)
