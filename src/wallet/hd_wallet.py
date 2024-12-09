import secrets
import hashlib

class HDWallet:
    def __init__(self, seed):
        self.seed = seed
        self.addresses = {}  # index -> (private_key, public_key)

    def generate_address(self, index):
        private_key = secrets.token_hex(32)  # 64 hex characters
        public_key = self._generate_public_key(private_key)
        self.addresses[index] = (private_key, public_key)
        return public_key

    def _generate_public_key(self, private_key):
        return hashlib.sha256(private_key.encode()).hexdigest()

    def get_address(self, index):
        if index not in self.addresses:
            return self.generate_address(index)
        return self.addresses[index][1]

    def export_wallet(self):
        return {
            "seed": self.seed,
            "addresses": self.addresses
        }

    @staticmethod
    def import_wallet(data):
        wallet = HDWallet(data['seed'])
        wallet.addresses = data['addresses']
        return wallet
