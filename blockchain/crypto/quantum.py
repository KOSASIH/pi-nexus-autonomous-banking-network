import numpy as np

class Lattice:
    def __init__(self, dimension: int, modulus: int):
        self.dimension = dimension
        self.modulus = modulus
        self.basis = np.random.randint(0, modulus, size=(dimension, dimension))

    def sample(self) -> np.ndarray:
        return np.random.randint(0, self.modulus, size=self.dimension)

    def encrypt(self, message: bytes) -> np.ndarray:
        message_int = int.from_bytes(message, byteorder='big')
        error = self.sample()
        ciphertext = (message_int + error) % self.modulus
        return ciphertext

    def decrypt(self, ciphertext: np.ndarray) -> bytes:
        error = self.sample()
        message_int = (ciphertext - error) % self.modulus
        return message_int.to_bytes((message_int.bit_length() + 7) // 8, byteorder='big')

class NewHope:
    def __init__(self, dimension: int, modulus: int):
        self.lattice = Lattice(dimension, modulus)

    def keygen(self) -> (np.ndarray, np.ndarray):
        sk = self.lattice.sample()
        pk = self.lattice.basis @ sk % self.lattice.modulus
        return pk, sk

    def encrypt(self, message: bytes, pk: np.ndarray) -> np.ndarray:
        return self.lattice.encrypt(message)

    def decrypt(self, ciphertext: np.ndarray, sk: np.ndarray) -> bytes:
        return self.lattice.decrypt(ciphertext)
