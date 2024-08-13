import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

class PostQuantumCryptography:
    def __init__(self, n, k, w, delta, q):
        self.n = n
        self.k = k
        self.w = w
        self.delta = delta
        self.q = q
        self.lattice = self.generate_lattice(n, q)
        self.code = self.generate_code(k, n)
        self.keypair = self.generate_keypair()

    def generate_lattice(self, n, q):
        lattice = np.random.randint(0, q, size=(n, n))
        return lattice

    def generate_code(self, k, n):
        code = np.random.randint(0, 2, size=(k, n))
        return code

    def generate_keypair(self):
        private_key = self.generate_private_key()
        public_key = self.generate_public_key(private_key)
        return private_key, public_key

    def generate_private_key(self):
        private_key = np.random.randint(0, self.q, size=self.k)
        return private_key

    def generate_public_key(self, private_key):
        public_key = np.dot(self.code, private_key) % self.q
        return public_key

    def keygen(self):
        return self.keypair

    def encrypt(self, message, public_key):
        ciphertext = np.dot(self.code, message) + self.generate_error_vector(self.n, self.w)
        return ciphertext

    def decrypt(self, ciphertext, private_key):
        message = np.dot(self.code, private_key) - self.generate_error_vector(self.n, self.w)
        return message

    def serialize_public_key(self, public_key):
        public_key = serialization.load_pem_public_key(
            b'-----BEGIN PUBLIC KEY-----\n' +
            b'-----END PUBLIC KEY-----\n',
            backend=default_backend()
        )
        public_key.public_numbers = utils.RSAPublicNumbers(
            e=public_key,
            n=self.n
        )
        return public_key

    def serialize_private_key(self, private_key):
        private_key = serialization.load_pem_private_key(
            b'-----BEGIN RSA PRIVATE KEY-----\n' +
            b'-----END RSA PRIVATE KEY-----\n',
            password=None,
            backend=default_backend()
        )
        private_key.private_numbers = utils.RSAPrivateNumbers(
            p=private_key,
            q=self.k,
            d=private_key,
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

    def generate_error_vector(self, n, w):
        error_vector = np.random.randint(0, 2, size=n)
        return error_vector

if __name__ == '__main__':
    n = 256
    k = 128
    w = 10
    delta = 0.5
    q = 12289
    pqc = PostQuantumCryptography(n, k, w, delta, q)
    keypair = pqc.keygen()
    public_key = pqc.serialize_public_key(keypair[1])
    private_key = pqc.serialize_private_key(keypair[0])
    message = b'Hello, world!'
    ciphertext = pqc.encrypt(message, public_key)
    decrypted_message = pqc.decrypt(ciphertext, private_key)
    print("Decrypted message:", decrypted_message)
    signature = pqc.sign(message, private_key)
    pqc.verify(message, signature, public_key)
    print("Signature verified successfully!")
