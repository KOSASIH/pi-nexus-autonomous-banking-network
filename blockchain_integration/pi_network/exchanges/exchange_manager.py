from typing import Dict, List, Union

import requests


class Exchange:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.example.com"

    def fetch_exchange_rates(self) -> Dict[str, float]:
        """Fetches the current exchange rates for the exchange."""
        pass

    def place_order(
        self, order: Dict[str, Union[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """Places an order on the exchange."""
        pass

    def get_account_balance(self) -> Dict[str, float]:
        """Fetches the current account balance for the exchange."""
        pass


class ExchangeManager:
    def __init__(self, exchanges: List[Exchange]):
        self.exchanges = exchanges

    def fetch_all_exchange_rates(self) -> Dict[str, Dict[str, float]]:
        """Fetches the current exchange rates for all exchanges."""
        exchange_rates = {}
        for exchange in self.exchanges:
            exchange_rates[exchange.name] = exchange.fetch_exchange_rates()
        return exchange_rates

    def place_order_on_all_exchanges(
        self, order: Dict[str, Union[str, float]]
    ) -> List[Dict[str, Union[str, float]]]:
        """Places an order on all exchanges."""
        orders = []
        for exchange in self.exchanges:
            orders.append(exchange.place_order(order))
        return orders

    def get_total_account_balance(self) -> float:
        """Fetches the total account balance across all exchanges."""
        total_balance = 0.0
        for exchange in self.exchanges:
            balance = exchange.get_account_balance()
            total_balance += sum(balance.values())
        return total_balance
