from abc import ABC, abstractmethod
from typing import List

class CrossChainBridge(ABC):
    def __init__(self, name: str, chain_id: int, contract_address: str):
        self.name = name
        self.chain_id = chain_id
        self.contract_address = contract_address

    @abstractmethod
    def bridge_tokens(self, tokens: List[str], amount: float) -> str:
        pass

    @abstractmethod
    def get_bridge_fee(self, tokens: List[str], amount: float) -> float:
        pass

class PiNetworkCrossChainBridge(CrossChainBridge):
    def __init__(self, name: str, chain_id: int, contract_address: str):
        super().__init__(name, chain_id, contract_address)

    def bridge_tokens(self, tokens: List[str], amount: float) -> str:
        # Implement Pi Network bridge logic
        pass

    def get_bridge_fee(self, tokens: List[str], amount: float) -> float:
        # Implement Pi Network bridge fee calculation
        pass

class EthereumCrossChainBridge(CrossChainBridge):
    def __init__(self, name: str, chain_id: int, contract_address: str):
        super().__init__(name, chain_id, contract_address)

    def bridge_tokens(self, tokens: List[str], amount: float) -> str:
        # Implement Ethereum bridge logic
        pass

    def get_bridge_fee(self, tokens: List[str], amount: float) -> float:
        # Implement Ethereum bridge fee calculation
        pass
