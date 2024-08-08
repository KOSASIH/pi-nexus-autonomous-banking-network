# Random oracle implementation for NTRU
import numpy as np

def random_oracle(size):
    # Generate random bytes for NTRU encryption
    return np.random.bytes(size)
