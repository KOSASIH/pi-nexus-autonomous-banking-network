import hashlib


class Blockchain:
    def __init__(self):
        self.chain = []

    def add_block(self, block: dict):
        self.chain.append(block)
        self.chain[-1]["hash"] = self._calculate_hash(self.chain[-1])

    def _calculate_hash(self, block: dict):
        return hashlib.sha256(str(block).encode()).hexdigest()
