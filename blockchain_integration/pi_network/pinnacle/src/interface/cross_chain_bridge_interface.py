from abc import ABC, abstractmethod
from typing import List

class CrossChainBridgeInterface(ABC):
    @abstractmethod
    def bridge_tokens(self, tokens: List[str], amount: float) -> str:
        """Bridge tokens from one chain to another"""
        pass

    @abstractmethod
    def get_bridge_fee(self, tokens: List[str], amount: float) -> float:
        """Get the bridge fee for a given token and amount"""
        pass

    @abstractmethod
    def get_supported_chains(self) -> List[str]:
        """Get the list of supported chains"""
        pass

    @abstractmethod
    def get_supported_tokens(self) -> List[str]:
        """Get the list of supported tokens"""
        pass
