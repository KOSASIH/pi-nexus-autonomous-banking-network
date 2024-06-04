# interoperability_protocol.py
import hashlib


class InteroperabilityProtocol:
    def __init__(self, blockchain_networks):
        self.blockchain_networks = blockchain_networks

    def send_asset(self, asset, from_network, to_network):
        # validate asset and networks
        asset_hash = hashlib.sha256(asset.encode()).hexdigest()
        self.blockchain_networks[from_network].send_transaction(asset_hash, to_network)

    def receive_asset(self, asset_hash, from_network):
        # validate asset hash and network
        self.blockchain_networks[from_network].receive_transaction(asset_hash)
