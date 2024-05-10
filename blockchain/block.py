import hashlib
import time


class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

    @staticmethod
    def calculate_hash(index, previous_hash, timestamp, data):
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def __str__(self):
        return "Block: {index}, Previous Hash: {previous_hash}, Timestamp: {timestamp}, Data: {data}, Hash: {hash}".format(
            index=self.index,
            previous_hash=self.previous_hash,
            timestamp=self.timestamp,
            data=self.data,
            hash=self.hash,
        )
