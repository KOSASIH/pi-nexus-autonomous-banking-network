import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

def generate_qr_keypair():
    private_key = ec.generate_private_key(
        ec.SECP256R1(),
        default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

def qrc_sign(transaction, private_key):
    signer = serialization.load_pem_private_key(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ),
        password=None,
        backend=default_backend()
    )
    signature = signer.sign(transaction.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    return signature

def qrc_verify(transaction, signature, public_key):
    verifier = serialization.load_pem_public_key(
        public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        ),
        backend=default_backend()
    )
    verifier.verify(signature, transaction.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    return True
