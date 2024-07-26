import hashlib
from typing import List, Dict

class NodeModel:
    def __init__(self, node_id: str, node_type: str, blockchain: 'BlockchainModel'):
        self.node_id = node_id
        self.node_type = node_type
        self.blockchain = blockchain
        self.peers = []

    def connect_peer(self, peer_node: 'NodeModel') -> None:
        # Connect to another node
        self.peers.append(peer_node)

    def disconnect_peer(self, peer_node: 'NodeModel') -> None:
        # Disconnect from another node
        self.peers.remove(peer_node)

    def sync_blockchain(self) -> None:
        # Synchronize blockchain with peers
        for peer in self.peers:
            self.blockchain.sync(peer.blockchain)

    def mine_block(self, block: Dict) -> None:
        # Mine a new block
        self.blockchain.add_block(block)

    def validate_block(self, block: Dict) -> bool:
        # Validate a block
        return self.blockchain.validate_block(block)

    def get_block(self, block_height: int) -> Dict:
        # Retrieve a block from the blockchain
        return self.blockchain.get_block(block_height)

    def get_transaction(self, transaction_id: str) -> Dict:
        # Retrieve a transaction from the blockchain
        return self.blockchain.get_transaction(transaction_id)
