import hashlib

class SHA3_256:
    def __init__(self):
        self.hash = hashlib.sha3_256()

    def update(self, data: bytes):
        self.hash.update(data)

    def digest(self) -> bytes:
        return self.hash.digest()

class BLAKE2b:
    def __init__(self):
        self.hash = hashlib.blake2b()

    def update(self, data: bytes):
        self.hash.update(data)

    def digest(self) -> bytes:
        return self.hash.digest()

class Argon2:
    def __init__(self, password: bytes, salt: bytes, memory_cost: int, parallelism: int):
        self.password = password
        self.salt = salt
        self.memory_cost = memory_cost
        self.parallelism = parallelism

    def hash(self) -> bytes:
        import argon2
        ph = argon2.PasswordHasher(memory_cost=self.memory_cost, parallelism=self.parallelism)
        return ph.hash(self.password, self.salt)
