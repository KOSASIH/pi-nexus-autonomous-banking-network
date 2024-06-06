# hdsb_blockchain.py
import numpy as np
from holographic_data_storage import HolographicDataStorage

class HDSB:
    def __init__(self):
        self.hds = HolographicDataStorage()

    def store_block(self, block_data):
        self.hds.store(block_data)

    def retrieve_block(self, block_id):
        block_data = self.hds.retrieve(block_id)
        return block_data

hdsb = HDSB()
