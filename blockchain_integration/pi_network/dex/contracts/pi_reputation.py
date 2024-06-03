# pi_reputation.py

import web3
from web3.contract import Contract

class PIReputation:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Reputation ABI from a file or database
        with open('pi_reputation.abi', 'r') as f:
            return json.load(f)

    def build_reputation(self, actions: list) -> bool:
        # Build reputation by performing actions on the network
        tx_hash = self.contract.functions.buildReputation(actions).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_reputation(self) -> int:
        # Get the current reputation of a user
        return self.contract.functions.getReputation().call()

    def set_reputation_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the reputation contract
        tx_hash = self.contract.functions.setReputationParameters(parameters).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
