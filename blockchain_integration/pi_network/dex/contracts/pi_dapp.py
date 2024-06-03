# pi_dapp.py

import web3
from pi_oracle import PIOracle
from pi_insurance import PIInsurance
from pi_nft import PINFT
from pi_exchange import PIExchange
from pi_bank import PIBank

class PIDApp:
    def __init__(self, web3: web3.Web3, contract_addresses: dict):
        self.web3 = web3
        self.oracle = PIOracle(web3, contract_addresses['oracle'])
        self.insurance = PIInsurance(web3, contract_addresses['insurance'])
        self.nft = PINFT(web3, contract_addresses['nft'])
        self.exchange = PIExchange(web3, contract_addresses['exchange'])
        self.bank = PIBank(web3, contract_addresses['bank'])

    def get_abi(self) -> dict:
        # Load the ABI for each contract
        abi = {
            'oracle': self.oracle.get_abi(),
            'insurance': self.insurance.get_abi(),
            'nft': self.nft.get_abi(),
            'exchange': self.exchange.get_abi(),
            'bank': self.bank.get_abi()
        }
        return abi

    def get_balance(self, asset: str) -> int:
        # Get the balance of an asset for a user
        return self.bank.get_balance(asset)

    def deposit(self, asset: str, amount: int) -> bool:
        # Deposit an asset into the bank
        return self.bank.deposit(asset, amount)

    def withdraw(self, asset: str, amount: int) -> bool:
        # Withdraw an asset from the bank
        return self.bank.withdraw(asset, amount)

    def create_oracle(self, name: str, metadata: str) -> bool:
        # Create a new oracle
        return self.oracle.create_oracle(name, metadata)

    def get_oracle(self, oracle_id: int) -> dict:
        # Get the details of an oracle
        return self.oracle.get_oracle(oracle_id)

    def update_oracle(self, oracle_id: int, metadata: str) -> bool:
        # Update the metadata of an oracle
        return self.oracle.update_oracle(oracle_id, metadata)

    def create_insurance(self, name: str, symbol: str, metadata: str) -> bool:
        # Create a new insurance contract
        return self.insurance.create_insurance(name, symbol, metadata)

    def get_insurance(self, contract_address: str) -> dict:
        # Get the details of an insurance contract
        return self.insurance.get_insurance(contract_address)

    def create_nft(self, name: str, symbol: str, metadata: str) -> bool:
        # Create a new NFT contract
        return self.nft.create_nft(name, symbol, metadata)

    def get_nft(self, contract_address: str) -> dict:
        # Get the details of an NFT contract
        return self.nft.get_nft(contract_address)

    def create_exchange(self, asset: str, exchange_parameters: dict) -> bool:
        # Create a new exchange contract for an asset
        return self.exchange.create_exchange(asset, exchange_parameters)

    def fill_order(self, order_id: int) -> bool:
        # Fill an order
        return self.exchange.fill_order(order_id)

    def get_order_book(self, asset: str) -> list:
        # Get the order book for an asset
        return self.exchange.get_order_book(asset)

    def set_exchange_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the exchange contract
        return self.exchange.set_exchange_parameters(parameters)

    def set_bank_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the bank contract
        return self.bank.set_bank_parameters(parameters)
