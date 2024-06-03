# pi_stablecoin.py

import web3
from web3.contract import Contract

class PISTablecoin:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Stablecoin ABI from a file or database
        with open('pi_stablecoin.abi', 'r') as f:
            return json.load(f)

    def mint(self, amount: int) -> bool:
        # Mint new stablecoins
        tx_hash = self.contract.functions.mint(amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def burn(self, amount: int) -> bool:
        # Burn existing stablecoins
        tx_hash = self.contract.functions.burn(amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_balance(self, address: str) -> int:
        # Get the balance of stablecoins for a user
        return self.contract.functions.getBalance(address).call()

    def get_total_supply(self) -> int:
        # Get the total supply of stablecoins
        return self.contract.functions.getTotalSupply().call()
