# tools/autonomous_global_exchanges_connector.py

import json
import logging
import os
from typing import Any, Dict, List

import requests


class Config:
    def __init__(self, config_file: str = "config.json"):
        self.config = self.load_config(config_file)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, "r") as f:
            return json.load(f)


class Exchange:
    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key

    def get_rates(self) -> List[Dict[str, str]]:
        response = requests.get(
            f"https://{self.name}.com/api/rates", headers={"API-KEY": self.api_key}
        )
        if response.status_code == 200:
            return response.json()["rates"]
        else:
            logging.error(f"Error fetching rates from {self.name}: {response.text}")
            return []

    def get_balances(self) -> List[Dict[str, str]]:
        response = requests.get(
            f"https://{self.name}.com/api/balances", headers={"API-KEY": self.api_key}
        )
        if response.status_code == 200:
            return response.json()["balances"]
        else:
            logging.error(f"Error fetching balances from {self.name}: {response.text}")
            return []

    def execute_trade(self, symbol: str, amount: float) -> bool:
        response = requests.post(
            f"https://{self.name}.com/api/trade",
            headers={"API-KEY": self.api_key},
            json={"symbol": symbol, "amount": amount},
        )
        if response.status_code == 200:
            return True
        else:
            logging.error(f"Error executing trade on {self.name}: {response.text}")
            return False


class AutonomousGlobalExchangesConnector:
    def __init__(self, config: Config):
        self.exchanges = [
            Exchange(name, api_key)
            for name, api_key in config.config["exchanges"].items()
        ]

    def get_exchange_rates(self) -> List[Dict[str, str]]:
        exchange_rates = []
        for exchange in self.exchanges:
            exchange_rates.extend(exchange.get_rates())
        return exchange_rates

    def get_account_balances(self) -> List[Dict[str, str]]:
        account_balances = []
        for exchange in self.exchanges:
            account_balances.extend(exchange.get_balances())
        return account_balances

    def execute_trade(self, exchange_name: str, symbol: str, amount: float) -> bool:
        exchange = next(
            (exchange for exchange in self.exchanges if exchange.name == exchange_name),
            None,
        )
        if exchange:
            return exchange.execute_trade(symbol, amount)
        else:
            logging.error(f"Exchange {exchange_name} not found")
            return False


if __name__ == "__main__":
    config = Config()
    connector = AutonomousGlobalExchangesConnector(config)
    exchange_rates = connector.get_exchange_rates()
    print(exchange_rates)
    account_balances = connector.get_account_balances()
    print(account_balances)
    connector.execute_trade("binance", "BTCUSDT", 0.01)
