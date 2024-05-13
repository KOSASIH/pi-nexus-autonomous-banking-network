import time
import uuid
import json
import requests
from typing import List, Dict, Any

class OffChain:
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the OffChain class with a dictionary of API keys.
        """
        self.api_keys = api_keys

    def get_data(self, url: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get data from an API endpoint.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['data_api']}"
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def post_data(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post data to an API endpoint.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['data_api']}"
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def get_random_id(self) -> str:
        """
        Generate a random ID.
        """
        return str(uuid.uuid4())

    def get_timestamp(self) -> int:
        """
Get the current timestamp.
        """
        return int(time.time())

    def get_weather(self, city: str) -> Dict[str, Any]:
        """
        Get the weather in a city.
        """
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": self.api_keys["weather_api"],
            "units": "metric"
        }
        return self.get_data(url, params)

    def get_exchange_rate(self, base: str, quote: str) -> float:
        """
        Get the exchange rate between two currencies.
        """
        url = "https://api.exchangerate-api.com/v4/latest/" + base
        params = {
            "symbols": quote
        }
        data = self.get_data(url, params)
        return data["rates"][quote]

    def get_gas_price(self) -> float:
        """
        Get the current gas price.
        """
        url = "https://ethgasstation.info/api/v3/gasprice/ethgasAPI"
        data = self.get_data(url)
        return data["fast"]

    def get_block_number(self) -> int:
        """
        Get the current block number.
        """
        web3 = Web3(Web3.HTTPProvider(self.api_keys["ethereum_node"]))
        return web3.eth.blockNumber

    def get_transaction_count(self, address: str) -> int:
        """
        Get the transaction count for an address.
        """
        web3 = Web3(Web3.HTTPProvider(self.api_keys["ethereum_node"]))
        return web3.eth.getTransactionCount(address)

    def get_balance(self, address: str) -> float:
        """
        Get the balance of an address.
        """
        web3 = Web3(Web3.HTTPProvider(self.api_keys["ethereum_node"]))
        balance = web3.eth.getBalance(address)
        return web3.fromWei(balance, "ether")

    def send_transaction(self, to: str, value: float, gas_price: float, gas_limit: int, nonce: int) -> str:
        """
        Send a transaction.
        """
        web3 = Web3(Web3.HTTPProvider(self.api_keys["ethereum_node"]))
        transaction = {
            "to": to,
            "value": web3.toWei(value, "ether"),
            "gasPrice": web3.toWei(gas_price, "gwei"),
            "gas": gas_limit,
            "nonce": nonce
        }
        signed_transaction = web3.eth.account.signTransaction(transaction, private_key=self.api_keys["ethereum_private_key"])
        tx_hash = web3.eth.sendRawTransaction(signed_transaction.rawTransaction)
        return web3.toHex(tx_hash)

    def call_contract_function(self, contract_address: str, abi: List[Dict[str, Any]], function_name: str, function_args: List[Any]) -> Any:
        """
        Call a function of a smart contract.
        """
        web3 = Web3(Web3.HTTPProvider(self.api_keys["ethereum_node"]))
        contract = web3.eth.contract(address=contract_address, abi=abi)
        return contract.functions[function_name](*function_args).call()

    def send_contract_transaction(self, contract_address: str, abi: List[Dict[str, Any]], function_name: str, function_args: List[Any], gas_price: float, gas_limit: int, nonce: int) -> str:
        """
        Send a transaction to a smart contract.
        """
        web3 = Web3(Web3.HTTPProvider(self.api_keys["ethereum_node"]))
        contract = web3.eth.contract(address=contract_address, abi=abi)
        transaction =contract.functions[function_name](*function_args).buildTransaction({
            "gasPrice": web3.toWei(gas_price, "gwei"),
            "gas": gas_limit,
            "nonce": nonce
        })
        signed_transaction = web3.eth.account.signTransaction(transaction, private_key=self.api_keys["ethereum_private_key"])
        tx_hash = web3.eth.sendRawTransaction(signed_transaction.rawTransaction)
        return web3.toHex(tx_hash)
