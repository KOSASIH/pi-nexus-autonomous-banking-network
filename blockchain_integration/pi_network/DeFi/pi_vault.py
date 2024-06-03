# pi_vault.py

import web3
from web3.contract import Contract

class PIVault:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Vault ABI from a file or database
        with open('pi_vault.abi', 'r') as f:
            return json.load(f)

    def deposit(self, asset: str, amount: int) -> bool:
        # Deposit an asset into the vault
        tx_hash = self.contract.functions.deposit(asset, amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def withdraw(self, asset: str, amount: int) -> bool:
        # Withdraw an asset from the vault
        tx_hash = self.contract.functions.withdraw(asset, amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_balance(self, asset: str) -> int:
        # Get the balance of an asset in the vault
        return self.contract.functions.getBalance(asset).call()
