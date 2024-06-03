# pi_staking.py

import web3
from web3.contract import Contract

class PISTaking:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI Staking ABI from a file or database
        with open('pi_staking.abi', 'r') as f:
            return json.load(f)

    def stake(self, amount: int) -> bool:
        # Stake PI tokens
        tx_hash = self.contract.functions.stake(amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def unstake(self, amount: int) -> bool:
        # Unstake PI tokens
        tx_hash = self.contract.functions.unstake(amount).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_staked_balance(self, address: str) -> int:
        # Get the staked balance of a user
        return self.contract.functions.getStakedBalance(address).call()

    def get_reward(self, address: str) -> int:
        # Get the reward for a user
        return self.contract.functions.getReward(address).call()
