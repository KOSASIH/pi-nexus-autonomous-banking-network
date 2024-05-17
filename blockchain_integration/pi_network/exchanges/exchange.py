# exchange.py
import json
from typing import Dict, List


class Exchange:
    def __init__(self, config: Dict):
        self.config = config
        self.coins = {}

    def list_coins(self) -> List[Dict]:
        """
        List all available Pi coins.

        Returns:
            List[Dict]: A list of dictionaries containing information about each coin.
        """
        return list(self.coins.values())

    def set_coin_value(self, coin: str, value: float) -> None:
        """
        Set the value of a Pi coin.

        Args:
            coin (str): The symbol of the coin.
            value (float): The value of the coin.
        """
        self.coins[coin] = {"symbol": coin, "value": value}

    def trade_coins(self, coin1: str, coin2: str, amount: float) -> None:
        """
        Trade one Pi coin for another.

        Args:
            coin1 (str): The symbol of the coin to trade.
            coin2 (str): The symbol of the coin to receive.
            amount (float): The amount of coin1 to trade.
        """
        coin1_value = self.coins[coin1]["value"]
        coin2_value = self.coins[coin2]["value"]

        self.coins[coin1]["value"] = coin1_value - amount
        self.coins[coin2]["value"] = coin2_value + amount

    def get_balance(self, coin: str) -> float:
        """
        Get the balance of a Pi coin.

        Args:
            coin (str): The symbol of the coin.

        Returns:
            float: The balance of the coin.
        """
        return self.coins[coin]["value"]

    def load_coins(self) -> None:
        """
        Load the available Pi coins from the configuration file.
        """
        with open(self.config["coins_file"], "r") as file:
            self.coins = json.load(file)

    def save_coins(self) -> None:
        """
        Save the available Pi coins to the configuration file.
        """
        with open(self.config["coins_file"], "w") as file:
            json.dump(self.coins, file)
