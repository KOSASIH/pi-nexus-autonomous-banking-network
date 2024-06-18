from stellar_sdk import Network

class StellarNetwork:
    def __init__(self, network_passphrase):
        self.network = Network.from_passphrase(network_passphrase)
