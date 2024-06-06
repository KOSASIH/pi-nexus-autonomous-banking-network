# fgbc_compression.py
import numpy as np
from fractal_geometry import FractalGeometry

class FGBC:
    def __init__(self):
        self.fg = FractalGeometry()

    def compress_data(self, blockchain_data):
        compressed_data = self.fg.compress(blockchain_data)
        return compressed_data

    def decompress_data(self, compressed_data):
        decompressed_data = self.fg.decompress(compressed_data)
        return decompressed_data

fgbc = FGBC()
