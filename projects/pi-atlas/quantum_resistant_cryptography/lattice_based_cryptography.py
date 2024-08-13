import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

class LatticeBasedCryptography:
    def __init__(self, n, q, sigma, alpha):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.alpha = alpha
        self.A = self.generate_matrix(n, q)
        self.s = self.generate_secret_key(n, q, sigma)
        self.e = self.generate_error_vector(n, q, alpha)

    def generate_matrix(self, n, q):
        A = np.random.randint(0, q, size=(n, n))
        return A

    def generate_secret_key(self, n, q, sigma):
        s = np.random.normal(0, sigma, size=n)
        s = np.mod(s, q)
        return s

    def generate_error_vector(self, n, q, alpha):
        e = np.random.normal(0, alpha, size=n)
        e = np.mod(e, q)
        return e

    def keygen(self):
        A_inv = np.linalg.inv(self.A)
        s_hat = np.dot(A_inv, self.s)
        e_hat = np.dot(A_inv, self.e)
        return s_hat, e_hat

    def encrypt(self, m):
        c = np.dot(self.A, m) + self.e
        return c

    def decrypt(self, c, s_hat):
        m = np.dot(s_hat, c) - self.e
        return m

    def serialize_public_key(self, s_hat, e_hat):
        public_key = serialization.load_pem_public_key(
            b'-----BEGIN PUBLIC KEY-----\n' +
            b'-----END PUBLIC KEY-----\n',
            backend=default_backend()
        )
        public_key.public_numbers = utils.RSAPublicNumbers(
            e=e_hat,
            n=s_hat
        )
        return public_key

    def serialize_private_key(self, s_hat, e_hat):
        private_key = serialization.load_pem_private_key(
            b'-----BEGIN RSA PRIVATE KEY-----\n' +
            b'-----END RSA PRIVATE KEY-----\n',
            password=None,
            backend=default_backend()
        )
        private_key.private_numbers = utils.RSAPrivateNumbers(
            p=s_hat,
            q=e_hat,
            d=s_hat,
            dp=e_hat,
            dq=e_hat,
            qi=e_hat
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
    q = 12289
    sigma = 3.2
    alpha = 2.5
    lbc = LatticeBasedCryptography(n, q, sigma, alpha)
    s_hat, e_hat = lbc.keygen()
    public_key = lbc.serialize_public_key(s_hat, e_hat)
    private_key = lbc.serialize_private_key(s_hat, e_hat)
    message = b'Hello, world!'
    signature = lbc.sign(message, private_key)
    lbc.verify(message, signature, public_key)
    print("Signature verified successfully!")
