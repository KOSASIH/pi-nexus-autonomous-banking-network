# Consensus Engine for Pi Network
import hashlib
import time
from pi_network.core.block import Block

class ConsensusEngine:
    def __init__(self, network_id, node_id):
        self.network_id = network_id
        self.node_id = node_id
        self.blockchain = []

    def verify_block(self, block: Block) -> bool:
        # Verify block hash, transactions, and signature
        if not self.verify_block_hash(block):
            return False
        if not self.verify_transactions(block):
            return False
        if not self.verify_signature(block):
            return False
        return True

    def verify_block_hash(self, block: Block) -> bool:
        # Verify block hash using SHA-256
        block_hash = hashlib.sha256(str(block).encode()).hexdigest()
        return block_hash == block.hash

    def verify_transactions(self, block: Block) -> bool:
        # Verify transactions using elliptic curve cryptography
        for tx in block.transactions:
            if not self.verify_transaction(tx):
                return False
        return True

    def verify_transaction(self, tx) -> bool:
        # Verify transaction signature using ECDSA
        public_key = tx.sender_public_key
        signature = tx.signature
        message = tx.message
        return self.ecdsa_verify(public_key, signature, message)

    def ecdsa_verify(self, public_key, signature, message) -> bool:
        # ECDSA verification using secp256k1 curve
        from ecdsa import VerifyingKey
        vk = VerifyingKey.from_string(public_key, curve=secp256k1)
        return vk.verify(signature, message.encode())

    def add_block(self, block: Block) -> bool:
        # Add block to blockchain if verified successfully
        if self.verify_block(block):
            self.blockchain.append(block)
            return True
        return False
