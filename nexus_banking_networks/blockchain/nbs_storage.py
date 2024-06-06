# nbs_storage.py
import numpy as np
from nanotechnology_storage import NanotechnologyStorage

class NBSS:
    def __init__(self):
        self.nbs = NanotechnologyStorage()

    def store_block(self, block):
        data = np.array(block, dtype=np.uint8)
        self.nbs.store(data)

    def retrieve_block(self, block_hash):
        data = self.nbs.retrieve(block_hash)
        return data

nbs_storage = NBSS()
