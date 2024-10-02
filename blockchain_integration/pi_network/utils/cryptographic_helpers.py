import base64
import hashlib
import hmac

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def generate_key_pair():
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    private_key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_key = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    )
    return private_key, public_key


def encrypt_data(data, public_key):
    public_key = serialization.load_ssh_public_key(
        public_key, backend=default_backend()
    )
    encrypted_data = public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return encrypted_data


def decrypt_data(encrypted_data, private_key):
    private_key = serialization.load_pem_private_key(
        private_key, password=None, backend=default_backend()
    )
    decrypted_data = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return decrypted_data


def sign_data(data, private_key):
    private_key = serialization.load_pem_private_key(
        private_key, password=None, backend=default_backend()
    )
    signer = private_key.signer(
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    signature = signer.sign(data)
    return signature


def verify_signature(data, signature, public_key):
    public_key = serialization.load_ssh_public_key(
        public_key, backend=default_backend()
    )
    verifier = public_key.verifier(
        signature,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    verifier.update(data)
    verifier.verify()


def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()


def hmac_sign(data, secret_key):
    return hmac.new(secret_key.encode(), data.encode(), hashlib.sha256).hexdigest()


def base64_encode(data):
    return base64.b64encode(data.encode()).decode()


def base64_decode(data):
    return base64.b64decode(data.encode()).decode()
