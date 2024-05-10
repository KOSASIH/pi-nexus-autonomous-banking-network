import hashlib

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 2

    def create_genesis_block(self):
        return Block(0, '0' * 64, int(time.time()), 'Genesis Block', self.calculate_hash(0, '0' * 64, int(time.time()), 'Genesis Block'))

    def calculate_hash(self, index, previous_hash, timestamp, data):
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    def is_chain_valid(self, chain):
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i - 1]

            if current_block.hash != self.calculate_hash(current_block.index, current_block.previous_hash, current_block.timestamp, current_block.data):
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def add_block(self, data):
        new_block = Block(len(self.chain), self.chain[-1].hash, int(time.time()), data, None)
        new_block.hash = self.calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.data)
        self.chain.append(new_block)

    def replace_chain(self, chain):
        if self.is_chain_valid(chain) and len(chain) > len(self.chain):
            self.chain = chain
