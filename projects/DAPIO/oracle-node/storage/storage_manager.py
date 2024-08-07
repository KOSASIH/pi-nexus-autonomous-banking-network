import os
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class StorageManager:
    def __init__(self, storage_path: str, private_key: str, public_key: str):
        self.storage_path = storage_path
        self.private_key = serialization.load_pem_private_key(private_key.encode(), password=None, backend=default_backend())
        self.public_key = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())

    def store_data(self, data: dict) -> None:
        encrypted_data = self.encrypt_data(data)
        with open(os.path.join(self.storage_path, "data.json"), "wb") as f:
            f.write(encrypted_data)

    def retrieve_data(self) -> dict:
        with open(os.path.join(self.storage_path, "data.json"), "rb") as f:
            encrypted_data = f.read()
        return self.decrypt_data(encrypted_data)

    def encrypt_data(self, data: dict) -> bytes:
        encrypted_data = self.public_key.encrypt(
            json.dumps(data).encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data

        def decrypt_data(self, encrypted_data: bytes) -> dict:
        decrypted_data = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return json.loads(decrypted_data.decode())
