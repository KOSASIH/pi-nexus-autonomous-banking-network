import hashlib
import time

class Block:
  def __init__(self, index, previous_hash, timestamp, data):
    self.index = index
    self.previous_hash = previous_hash
    self.timestamp = timestamp
    self.data = data
    self.hash = self.calculate_hash()

  def calculate_hash(self):
    data_string = str(self.index) + self.previous_hash + str(self.timestamp) + str(self.data)
    return hashlib.sha256(data_string.encode()).hexdigest()

class Blockchain:
  def __init__(self):
    self.chain = [self.create_genesis_block()]

  def create_genesis_block(self):
    return Block(0, "0", int(time.time()), "Genesis Block")

  def get_latest_block(self):
    return self.chain[-1]

  def add_block(self, new_block):
    new_block.previous_hash = self.get_latest_block().hash
    new_block.hash = new_block.calculate_hash()
    self.chain.append(new_block)
