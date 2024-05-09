from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Cryptography:
    @staticmethod
    def generate_key_pair():
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        public_key = private_key.public_key()

        return private_key, public_key

    @staticmethod
    def save_private_key(private_key, filename):
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )

        with open(filename, 'wb') as f:
            f.write(pem)

    @staticmethod
    def load_private_key(filename):
        with open(filename, 'rb') as f:
            pem = f.read()

        private_key = serialization.load_pem_private_key(
            pem,
            password=None,
            backend=default_backend()
        )

        return private_key

    @staticmethod
    def save_public_key(public_key, filename):
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        with open(filename, 'wb') as f:
            f.write(pem)

    @staticmethod
    def load_public_key(filename):
        with open(filename, 'rb') as f:
            pem = f.read()

        public_key = serialization.load_pem_public_key(
            pem,
            backend=default_backend()
        )

        return public_key

    @staticmethod
    def encrypt_message(message, public_key):
        ciphertext = public_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return ciphertext

    @staticmethod
    def decrypt_message(ciphertext, private_key):
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
algorithm=hashes.SHA256(),
                label=None
            )
        )

        return plaintext

    @staticmethod
    def encrypt_symmetric(message, key):
        nonce = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag_length=16))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(message) + encryptor.finalize()
        tag = encryptor.tag

        return ciphertext, tag, nonce

    @staticmethod
    def decrypt_symmetric(ciphertext, tag, nonce, key):
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag_length=16))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    @staticmethod
    def derive_key(password, salt):
        kdf = HKDF(
            algorithm=SHA256(),
            length=32,
            salt=salt,
            info=None,
            backend=default_backend()
        )

        key = kdf.derive(password)

        return key

    @staticmethod
    def generate_hmac(message, key):
        hmac = HMAC(key, SHA256(), backend=default_backend())
        hmac.update(message)

        return hmac.finalize()
