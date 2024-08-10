from models import Blockchain

class BlockchainService:
  def __init__(self):
    self.blockchain = Blockchain()

  def get_blockchain(self):
    return self.blockchain.chain

  def add_block(self, new_block):
    self.blockchain.add_block(new_block)
    return self.blockchain.chain

  def validate_blockchain(self):
    for i in range(1, len(self.blockchain.chain)):
      current_block = self.blockchain.chain[i]
      previous_block = self.blockchain.chain[i - 1]
      if current_block.hash != current_block.calculate_hash():
        return False
      if current_block.previous_hash != previous_block.hash:
        return False
    return True
