# Multi-Signature Wallet for Pi Network
import hashlib
from pi_network.core.wallet import Wallet

class MultiSignatureWallet:
    def __init__(self, network_id, node_id):
        self.network_id = network_id
        self.node_id = node_id
        self.wallet_registry = {}

    def create_wallet(self, public_keys: list, threshold: int) -> str:
        # Create new multi-signature wallet and return wallet ID
        wallet_hash = hashlib.sha256("".join(public_keys).encode()).hexdigest()
        self.wallet_registry[wallet_hash] = Wallet(public_keys, threshold)
        return wallet_hash

    def get_wallet(self, wallet_id: str) -> Wallet:
        # Return wallet by ID
        return self.wallet_registry.get(wallet_id)

    def sign_transaction(self, wallet_id: str, transaction: dict) -> bool:
        # Sign transaction with multi-signature wallet
        wallet = self.wallet_registry.get(wallet_id)
        if wallet:
            return wallet.sign_transaction(transaction)
        return False
