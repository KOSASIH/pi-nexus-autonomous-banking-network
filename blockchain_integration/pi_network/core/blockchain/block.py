import hashlib
import time


class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

    def calculate_hash(self):
        value = (
            str(self.index) + self.previous_hash + str(self.timestamp) + str(self.data)
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(
            0, "0" * 64, int(time.time()), "Genesis Block", self.calculate_hash()
        )

    def add_block(self, data):
        previous_block = self.chain[-1]
        new_block = Block(
            len(self.chain), previous_block.hash, int(time.time()), data, None
        )
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True
