from typing import List
from models.cross_chain_bridge import CrossChainBridge

class CrossChainService:
    def __init__(self, cross_chain_bridges: List[CrossChainBridge]):
        self.cross_chain_bridges = cross_chain_bridges

    def bridge_tokens(self, tokens: List[str], amount: float) -> str:
        for bridge in self.cross_chain_bridges:
            transaction_id = bridge.bridge_tokens(tokens, amount)
            if transaction_id:
                return transaction_id
        return None

    def get_bridge_fee(self, tokens: List[str], amount: float) -> float:
        fees = []
        for bridge in self.cross_chain_bridges:
            fees.append(bridge.get_bridge_fee(tokens, amount))
        return sum(fees)
