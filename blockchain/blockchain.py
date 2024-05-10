import hashlib
import time


class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 5

    def create_genesis_block(self):
        return Block(
            0,
            "0" * 64,
            int(time.time()),
            "Genesis Block",
            self.calculate_hash(0, "0" * 64, int(time.time()), "Genesis Block"),
        )

    def calculate_hash(self, index, previous_hash, timestamp, data):
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def add_block(self, block):
        block.previous_hash = self.chain[-1].hash
        block.hash = self.calculate_hash(
            block.index, block.previous_hash, block.timestamp, block.data
        )
        self.chain.append(block)

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != self.calculate_hash(
                current_block.index,
                current_block.previous_hash,
                current_block.timestamp,
                current_block.data,
            ):
                return False

            if previous_block.hash != current_block.previous_hash:
                return False

        return True
