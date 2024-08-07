from typing import List
from models.liquidity_provider import LiquidityProvider

class LiquidityService:
    def __init__(self, liquidity_providers: List[LiquidityProvider]):
        self.liquidity_providers = liquidity_providers

    def get_liquidity(self, token: str) -> float:
        liquidity = 0
        for provider in self.liquidity_providers:
            liquidity += provider.get_liquidity(token)
        return liquidity

    def place_order(self, token: str, amount: float, price: float) -> str:
        for provider in self.liquidity_providers:
            order_id = provider.place_order(token, amount, price)
            if order_id:
                return order_id
        return None

    def cancel_order(self, order_id: str) -> bool:
        for provider in self.liquidity_providers:
            if provider.cancel_order(order_id):
                return True
        return False
