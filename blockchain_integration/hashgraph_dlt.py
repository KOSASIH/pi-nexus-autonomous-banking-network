import hashlib
from hashgraph import Hashgraph

class HashgraphDLT:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.hashgraph = Hashgraph()

    def add_node(self, node: str):
        self.nodes.append(node)
        self.hashgraph.add_node(node)

    def create_transaction(self, sender: str, recipient: str, amount: int) -> str:
        transaction = f"{sender}->{recipient}:{amount}"
        transaction_hash = hashlib.sha256(transaction.encode()).hexdigest()
        self.hashgraph.add_transaction(transaction_hash)
        return transaction_hash

    def validate_transaction(self, transaction_hash: str) -> bool:
        return self.hashgraph.validate_transaction(transaction_hash)

    def get_ledger(self) -> list:
        return self.hashgraph.get_ledger()
