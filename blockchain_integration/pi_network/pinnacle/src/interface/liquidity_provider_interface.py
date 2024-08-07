from abc import ABC, abstractmethod
from typing import List

class LiquidityProviderInterface(ABC):
    @abstractmethod
    def get_liquidity(self, token: str) -> float:
        """Get the available liquidity for a given token"""
        pass

    @abstractmethod
    def place_order(self, token: str, amount: float, price: float) -> str:
        """Place an order on the liquidity provider"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on the liquidity provider"""
        pass

    @abstractmethod
    def get_order_book(self, token: str) -> List[dict]:
        """Get the order book for a given token"""
        pass

    @abstractmethod
    def get_trading_pairs(self) -> List[str]:
        """Get the list of available trading pairs"""
        pass
