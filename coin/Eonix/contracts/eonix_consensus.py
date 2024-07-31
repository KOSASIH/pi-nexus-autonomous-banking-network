# eonix_consensus.py
import hashlib
import json
from eonix_network import EonixNetwork

class EonixConsensus:
    def __init__(self, network: EonixNetwork):
        self.network = network

    def resolve_conflict(self, new_chain):
        if new_chain is None or not self.is_valid_chain(new_chain):
            return False
        if len(new_chain) <= len(self.network.get_chain()):
            return False
        self.network.chain = new_chain
        return True

    def is_valid_chain(self, chain):
        for i in range(1, len(chain)):
            block = chain[i]
            previous_block = chain[i - 1]
            if block.get_previous_block_hash() != previous_block.get_block_hash():
                return False
            if not block.validate():
                return False
        return True

    def get_latest_block(self):
        return self.network.get_chain()[-1]

    def get_chain_length(self):
        return len(self.network.get_chain())

    def to_dict(self):
        return {
            "network": self.network.to_dict()
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, consensus_dict, network: EonixNetwork):
        consensus = cls(network)
        return consensus

    @classmethod
    def from_json(cls, consensus_json, network: EonixNetwork):
        consensus_dict = json.loads(consensus_json)
        return cls.from_dict(consensus_dict, network)
