# pi_exchange.py

import web3
from web3.contract import Contract


class PIExchange:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=self.get_abi()
        )

    def get_abi(self) -> list:
        # Load the PI Exchange ABI from a file or database
        with open("pi_exchange.abi", "r") as f:
            return json.load(f)

    def create_order(self, asset: str, amount: int, price: int) -> bool:
        # Create a new order to buy or sell an asset
        tx_hash = self.contract.functions.createOrder(asset, amount, price).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def cancel_order(self, order_id: int) -> bool:
        # Cancel an order
        tx_hash = self.contract.functions.cancelOrder(order_id).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def fill_order(self, order_id: int) -> bool:
        # Fill an order
        tx_hash = self.contract.functions.fillOrder(order_id).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def get_order_book(self, asset: str) -> list:
        # Get the order book for an asset
        return self.contract.functions.getOrderBook(asset).call()

    def set_exchange_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the exchange contract
        tx_hash = self.contract.functions.setExchangeParameters(parameters).transact(
            {"from": self.web3.eth.accounts[0]}
        )
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
