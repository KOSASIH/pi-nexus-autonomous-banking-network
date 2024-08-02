# pi_smart_contract.py

import web3
from web3.contract import Contract


class PISmartContract:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI Smart Contract ABI from a file or database
        with open("pi_smart_contract.abi", "r") as f:
            return json.load(f)

    def create_contract(self, bytecode: str, abi: list) -> bool:
        # Create a new smart contract
        tx_hash = self.contract.functions.createContract(bytecode, abi).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def call_contract_function(
        self, contract_address: str, function_name: str, args: list
    ) -> any:
        # Call a function on a smart contract
        contract = self.web3.eth.contract(address=contract_address, abi=self.get_abi())
        return contract.functions[function_name](*args).call()

    def send_transaction_to_contract(
        self, contract_address: str, function_name: str, args: list
    ) -> bool:
        # Send a transaction to a smart contract
        contract = self.web3.eth.contract(address=contract_address, abi=self.get_abi())
        tx_hash = contract.functions[function_name](*args).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_contract_events(self, contract_address: str) -> list:
        # Get the events for a smart contract
        contract = self.web3.eth.contract(address=contract_address, abi=self.get_abi())
        return contract.events.getLogs()

    def set_contract_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the smart contract
        tx_hash = self.contract.functions.setParameters(parameters).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
