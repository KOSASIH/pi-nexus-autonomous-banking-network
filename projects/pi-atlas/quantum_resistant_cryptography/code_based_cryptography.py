import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

class CodeBasedCryptography:
    def __init__(self, n, k, w, delta):
        self.n = n
        self.k = k
        self.w = w
        self.delta = delta
        self.G = self.generate_generator_matrix(n, k)
        self.H = self.generateparity_check_matrix(n, k)
        self.x = self.generate_secret_key(k)
        self.e = self.generate_error_vector(n, w)

    def generate_generator_matrix(self, n, k):
        G = np.random.randint(0, 2, size=(n, k))
        return G

    def generate_parity_check_matrix(self, n, k):
        H = np.random.randint(0, 2, size=(n-k, n))
        return H

    def generate_secret_key(self, k):
        x = np.random.randint(0, 2, size=k)
        return x

    def generate_error_vector(self, n, w):
        e = np.random.randint(0, 2, size=n)
        e = np.mod(e, 2)
        return e

    def keygen(self):
        y = np.dot(self.G, self.x) + self.e
        return y

    def encrypt(self, m):
        c = np.dot(self.G, m) + self.e
        return c

    def decrypt(self, c, x):
        m = np.dot(self.G, x) - self.e
        return m

    def syndrome_decoding(self, c):
        s = np.dot(self.H, c)
        return s

    def error_correction(self, s):
        e = self.syndrome_decoding(s)
        return e

    def serialize_public_key(self, y):
        public_key = serialization.load_pem_public_key(
            b'-----BEGIN PUBLIC KEY-----\n' +
            b'-----END PUBLIC KEY-----\n',
            backend=default_backend()
        )
        public_key.public_numbers = utils.RSAPublicNumbers(
            e=y,
            n=self.n
        )
        return public_key

    def serialize_private_key(self, x):
        private_key = serialization.load_pem_private_key(
            b'-----BEGIN RSA PRIVATE KEY-----\n' +
            b'-----END RSA PRIVATE KEY-----\n',
            password=None,
            backend=default_backend()
        )
        private_key.private_numbers = utils.RSAPrivateNumbers(
            p=x,
            q=self.k,
            d=x,
            dp=self.k,
            dq=self.k,
            qi=self.k
        )
        return private_key

    def sign(self, message, private_key):
        signer = padding.PSS(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        )
        signature = private_key.sign(
            message,
            padding=signer,
            algorithm=hashes.SHA256()
        )
        return signature

    def verify(self, message, signature, public_key):
        verifier = padding.PSS(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        )
        public_key.verify(
            signature,
            message,
            padding=verifier,
            algorithm=hashes.SHA256()
        )

if __name__ == '__main__':
    n = 256
    k = 128
    w = 10
    delta = 0.5
    cbc = CodeBasedCryptography(n, k, w, delta)
    y = cbc.keygen()
    public_key = cbc.serialize_public_key(y)
    private_key = cbc.serialize_private_key(y)
    message = b'Hello, world!'
    signature = cbc.sign(message, private_key)
    cbc.verify(message, signature, public_key)
    print("Signature verified successfully!")
