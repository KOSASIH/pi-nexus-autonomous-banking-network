import hashlib
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class ConsensusAlgorithm:
    def __init__(self, nodes: list, threshold: int):
        self.nodes = nodes
        self.threshold = threshold
        self.public_keys = [serialization.load_pem_public_key(node.encode(), backend=default_backend()) for node in nodes]

    def propose_block(self, block: dict) -> bytes:
        proposal = json.dumps(block).encode()
        signature = ec.ECDSA(self.public_keys[0]).sign(proposal, hashlib.sha256(proposal).digest())
        return proposal + signature

    def verify_proposal(self, proposal: bytes) -> bool:
        proposal_data, signature = proposal[:-64], proposal[-64:]
        for public_key in self.public_keys:
            try:
                ec.ECDSA(public_key).verify(signature, hashlib.sha256(proposal_data).digest())
                return True
            except:
                pass
        return False

    def finalize_block(self, block: dict) -> bytes:
        finalized_block = json.dumps(block).encode()
        return finalized_block

    def run_consensus(self) -> bytes:
        while True:
            proposal = self.propose_block({"data": "example_data"})
            if self.verify_proposal(proposal):
                return self.finalize_block({"data": "example_data"})
            time.sleep(1)
