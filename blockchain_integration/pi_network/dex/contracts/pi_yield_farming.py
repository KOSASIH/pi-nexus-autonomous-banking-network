# pi_yield_farming.py

import web3
from web3.contract import Contract

class PIYieldFarming:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Yield Farming ABI from a file or database
        with open('pi_yield_farming.abi', 'r') as f:
            return json.load(f)

    def deposit(self, asset: str, amount: int) -> bool:
        # Deposit liquidity into a yield farming pool
        tx_hash = self.contract.functions.deposit(asset, amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def withdraw(self, asset: str, amount: int) -> bool:
        # Withdraw liquidity from a yield farming pool
        tx_hash = self.contract.functions.withdraw(asset, amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def claim_rewards(self) -> bool:
        # Claim rewards for providing liquidity
        tx_hash = self.contract.functions.claimRewards().transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_pool_list(self) -> list:
        # Get the list of yield farming pools
        return self.contract.functions.getPoolList().call()

    def get_user_rewards(self, user: str) -> int:
        # Get the user's rewards
        return self.contract.functions.getUserRewards(user).call()
