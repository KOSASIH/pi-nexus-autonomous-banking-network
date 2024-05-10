import time

from blockchain.block import Blockchain


class Miner:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def mine_block(self, data):
        while True:
            if self.blockchain.is_chain_valid(self.blockchain.chain):
                self.blockchain.add_block(data)
                print("Block successfully mined!")
                print("Proof of work: {}".format(self.blockchain.chain[-1].hash))
                break
            else:
                print("Block not mined. Invalid chain.")
                self.blockchain.chain = [self.blockchain.create_genesis_block()]
                self.blockchain.difficulty += 1
                print("New difficulty: {}".format(self.blockchain.difficulty))
                time.sleep(1)
