import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

class QSB:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.blockchain = []

    def create_block(self, transactions):
        # Create a new block with the given transactions
        block = {"transactions": transactions, "previous_hash": self.get_previous_hash()}
        self.blockchain.append(block)
        return block

    def get_previous_hash(self):
        # Get the hash of the previous block
        if not self.blockchain:
            return "0" * 64
        return hashlib.sha256(str(self.blockchain[-1]).encode()).hexdigest()

    def verify_block(self, block):
        # Verify the block using a quantum-secured digital signature scheme
        signature = block["signature"]
        message = str(block["transactions"]) + block["previous_hash"]
        public_key = self.get_public_key()
        verifier = public_key.verifier(
            signature,
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        try:
            verifier.verify(message.encode())
            return True
        except InvalidSignature:
            return False

    def get_public_key(self):
        # Get the public key of the node
        return self.nodes[0].public_key

nodes = [Node() for _ in range(5)]
qsb = QSB(num_nodes=5)
transactions = [{"from": "Alice", "to": "Bob", "amount": 10}, {"from": "Bob", "to": "Charlie", "amount": 5}]
block = qsb.create_block(transactions)
print("Block created:", block)
