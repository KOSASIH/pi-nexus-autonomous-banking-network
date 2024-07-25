import cryptography
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class SidraChainDataSecurity:
    def __init__(self, connector):
        self.connector = connector

    def generate_key_pair(self):
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        private_key = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )
        return private_key, public_key

    def encrypt_data(self, dataset_id, data):
        data_security = self.connector.get_data_security(dataset_id)
        public_key = serialization.load_ssh_public_key(data_security['public_key'], backend=default_backend())
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data

    def decrypt_data(self, dataset_id, encrypted_data):
        data_security = self.connector.get_data_security(dataset_id)
        private_key = serialization.load_pem_private_key(data_security['private_key'], password=None, backend=default_backend())
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data
