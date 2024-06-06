# hds_data_storage.py
import numpy as np
from holographic_data_storage import HolographicDataStorage

class HDS:
    def __init__(self):
        self.hds = HolographicDataStorage()

    def store_data(self, data):
        self.hds.store(data)

    def retrieve_data(self, data_id):
        data = self.hds.retrieve(data_id)
        return data

hds = HDS()
