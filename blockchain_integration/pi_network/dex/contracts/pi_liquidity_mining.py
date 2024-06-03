# pi_liquidity_mining.py

import web3
from web3.contract import Contract


class PILiquidityMining:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI Liquidity Mining ABI from a file or database
        with open("pi_liquidity_mining.abi", "r") as f:
            return json.load(f)

    def add_liquidity(self, asset: str, amount: int) -> bool:
        # Add liquidity to the network
        tx_hash = self.contract.functions.addLiquidity(asset, amount).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def remove_liquidity(self, asset: str, amount: int) -> bool:
        # Remove liquidity from the network
        tx_hash = self.contract.functions.removeLiquidity(asset, amount).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_liquidity_balance(self, asset: str) -> int:
        # Get the user's liquidity balance
        return self.contract.functions.getLiquidityBalance(asset).call()

    def set_liquidity_parameters(self, asset: str, parameters: dict) -> bool:
        # Set the liquidity parameters for an asset
        tx_hash = self.contract.functions.setLiquidityParameters(
            asset, parameters
        ).transact({"from": self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
