import numpy as np

class Paillier:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.n = p * q
        self.n_squared = self.n ** 2
        self.g = self.n + 1
        self.lambda_ = (p - 1) * (q - 1)

    def encrypt(self, plaintext):
        r = np.random.randint(1, self.n)
        ciphertext = (pow(self.g, plaintext, self.n_squared) * pow(r, self.n, self.n_squared)) % self.n_squared
        return ciphertext

    def decrypt(self, ciphertext):
        u = pow(ciphertext, self.lambda_, self.n_squared)
        plaintext = (u - 1) // self.n % self.n
        return plaintext

    def add_encrypted(self, c1, c2):
        return (c1 * c2) % self.n_squared

# Example usage
if __name__ == "__main__":
    p = 11
    q = 13
    paillier = Paillier(p, q)

    plaintext1 = 5
    plaintext2 = 7

    encrypted1 = paillier.encrypt(plaintext1)
    encrypted2 = paillier.encrypt(plaintext2)

    encrypted_sum = paillier.add_encrypted(encrypted1, encrypted2)
    decrypted_sum = paillier.decrypt(encrypted_sum)

    print(f"Encrypted 1: {encrypted1}")
    print(f"Encrypted 2: {encrypted2}")
    print(f"Decrypted Sum: {decrypted_sum}")
