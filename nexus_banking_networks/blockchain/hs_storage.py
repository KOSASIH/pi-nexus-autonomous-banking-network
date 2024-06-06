# hs_storage.py
import numpy as np
from holographic_storage import HolographicStorage

class HSStorage:
    def __init__(self):
        self.hs = HolographicStorage()

    def store_block(self, block):
        data = np.array(block, dtype=np.uint8)
        self.hs.store(data)

    def retrieve_block(self, block_hash):
        data = self.hs.retrieve(block_hash)
        return data

hs_storage = HSStorage()
