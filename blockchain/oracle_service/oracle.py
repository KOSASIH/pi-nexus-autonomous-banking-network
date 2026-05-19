import json

import requests
from web3 import Web3


class Oracle:
    def __init__(self, rpc_url):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))

    def get_exchange_rate(self, base, quote):
        """Get the exchange rate from the decentralized oracle contract."""
        # Assuming the oracle contract is deployed at 0x1234567890123456789012345678901234567890
        oracle_contract_address = Web3.toChecksumAddress(
            "0x1234567890123456789012345678901234567890"
        )
        exchange_rate_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "_base", "type": "address"},
                    {"name": "_quote", "type": "address"},
                ],
                "name": "getExchangeRate",
                "outputs": [{"name": "", "type": "uint256"}],
                "payable": False,
                "stateMutability": "view",
                "type": "function",
            }
        ]
        oracle_contract = self.web3.eth.contract(
            address=oracle_contract_address, abi=exchange_rate_abi
        )
        exchange_rate = oracle_contract.functions.getExchangeRate(base, quote).call()
        return exchange_rate

    def get_price(self, symbol):
        """Get the price of a cryptocurrency from a price API."""
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            price = data[list(data.keys())[0]]["usd"]
            return price
        else:
            raise Exception("Failed to get price from API")

    def get_weather(self, city):
        """Get the weather in a city from a weather API."""
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=YOUR_API_KEY"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "city": data["name"],
                "country": data["sys"]["country"],
                # Convert from Kelvin to Celsius
                "temperature": data["main"]["temp"] - 273.15,
                "description": data["weather"][0]["description"],
            }
            return weather
        else:
            raise Exception("Failed to get weather from API")
