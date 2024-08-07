import threading
from .consensus.consensus_algorithm import ConsensusAlgorithm
from .p2p_networking.p2p_node import P2PNode
from .storage.storage_manager import StorageManager

class OracleNode:
    def __init__(self, host: str, port: int, private_key: str, public_key: str, storage_path: str):
        self.host = host
        self.port = port
        self.private_key = private_key
        self.public_key = public_key
        self.storage_path = storage_path
        self.consensus_algorithm = ConsensusAlgorithm(["node1", "node2", "node3"], 2)
        self.p2p_node = P2PNode(host, port, private_key, public_key)
        self.storage_manager = StorageManager(storage_path, private_key, public_key)

    def start_node(self) -> None:
        threading.Thread(target=self.p2p_node.connect, args=("node2",)).start()
        threading.Thread(target=self.p2p_node.connect, args=("node3",)).start()
        self.consensus_algorithm.run_consensus()

    def receive_data(self, data: dict) -> None:
        self.storage_manager.store_data(data)

    def send_data(self, data: dict) -> None:
        self.p2p_node.send_message(json.dumps(data))
