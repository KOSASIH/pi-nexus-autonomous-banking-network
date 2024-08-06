from .quantum import NewHope
from .hash import SHA3_256, BLAKE2b, Argon2

def new_hope_keygen(dimension: int, modulus: int) -> (np.ndarray, np.ndarray):
    return NewHope(dimension, modulus).keygen()

def sha3_256(data: bytes) -> bytes:
    sha3 = SHA3_256()
    sha3.update(data)
    return sha3.digest()

def blake2b(data: bytes) -> bytes:
    blake2 = BLAKE2b()
    blake2.update(data)
    return blake2.digest()

def argon2(password: bytes, salt: bytes, memory_cost: int, parallelism: int) -> bytes:
    return Argon2(password, salt, memory_cost, parallelism).hash()
