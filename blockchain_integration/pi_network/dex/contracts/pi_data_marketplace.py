# pi_data_marketplace.py

import web3
from web3.contract import Contract


class PIDataMarketplace:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI Data Marketplace ABI from a file or database
        with open("pi_data_marketplace.abi", "r") as f:
            return json.load(f)

    def list_data(self, data: dict) -> bool:
        # List data for sale on the marketplace
        tx_hash = self.contract.functions.listData(data).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def buy_data(self, data_id: int) -> bool:
        # Buy data from the marketplace
        tx_hash = self.contract.functions.buyData(data_id).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_data(self, data_id: int) -> dict:
        # Get the details of a data listing
        return self.contract.functions.getData(data_id).call()

    def set_marketplace_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the data marketplace
        tx_hash = self.contract.functions.setMarketplaceParameters(parameters).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
