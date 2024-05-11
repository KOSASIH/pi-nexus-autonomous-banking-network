import random
import numpy as np

class HomomorphicEncryption:
    def __init__(self, key_size=1024):
        self.key_size = key_size

    def generate_keys(self):
        """
        Generates a pair of public and private keys for homomorphic encryption.
        """
        p = 2 * self.key_size + 1
        q = 2 * self.key_size + 1
        n = p * q

        g = random.randint(2, n - 1)
        while gcd(g, n) != 1:
            g = random.randint(2, n - 1)

        lambda_ = (p - 1) * (q - 1)
        mu = pow(lambda_, -1, n)

        public_key = (n, g)
        private_key = (lambda_, mu)

        return public_key, private_key

    def encrypt(self, public_key, value):
        """
        Encrypts a value using the public key.
        """
        n, g = public_key
        r = random.randint(2, n - 1)
        c = pow(g, value, n) * pow(r, n, n) % n

        return c

    def decrypt(self, private_key, ciphertext):
        """
        Decrypts a ciphertext using the private key.
        """
        lambda_, mu = private_key
        n, _ = public_key

        m = pow(ciphertext, lambda_, n) * pow(pow(r, lambda_, n), -1, n) % n

        return m

    def add(self, public_key, ciphertext1, ciphertext2):
        """
        Adds two ciphertexts using the public key.
        """
        n, _ = public_key

        ciphertext3 = (ciphertext1 * ciphertext2) % n

        return ciphertext3

    def multiply(self, public_key, ciphertext, value):
        """
        Multiplies a ciphertext by a value using the public key.
        """
        n, _ = public_key

        ciphertext2 = pow(ciphertext, value, n)

        return ciphertext2
