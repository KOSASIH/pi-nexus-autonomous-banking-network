import ipfshttpclient
import json
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class IPFSStorage:
    def __init__(self, ipfs_api_url, ipfs_api_port, private_key_path):
        self.ipfs_client = ipfshttpclient.connect(f"/ip4/{ipfs_api_url}/tcp/{ipfs_api_port}/http")
        self.private_key = self._load_private_key(private_key_path)
        self.data_cache = {}

    async def store_data(self, data, encryption_key):
        # Encrypt data using the provided encryption key
        encrypted_data = self._encrypt_data(data, encryption_key)

        # Add data to IPFS
        ipfs_hash = await self.ipfs_client.add(encrypted_data)

        # Store IPFS hash in cache
        self.data_cache[ipfs_hash] = encrypted_data

        return ipfs_hash

    async def retrieve_data(self, ipfs_hash, encryption_key):
        # Check if data is cached
        if ipfs_hash in self.data_cache:
            encrypted_data = self.data_cache[ipfs_hash]
        else:
            # Retrieve data from IPFS
            encrypted_data = await self.ipfs_client.cat(ipfs_hash)

        # Decrypt data using the provided encryption key
        decrypted_data = self._decrypt_data(encrypted_data, encryption_key)

        return decrypted_data

    def _load_private_key(self, private_key_path):
        with open(private_key_path, "rb") as f:
            private_key_pem = f.read()
        private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())
        return private_key

    def _encrypt_data(self, data, encryption_key):
        # Encrypt data using RSA-OAEP
        encrypted_data = self.private_key.encrypt(
            json.dumps(data).encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data

    def _decrypt_data(self, encrypted_data, encryption_key):
        # Decrypt data using RSA-OAEP
        decrypted_data = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return json.loads(decrypted_data.decode())

# Example usage
ipfs_storage = IPFSStorage("localhost", 5001, "path/to/private/key")
data = {"hello": "world"}
encryption_key = "my_secret_key"
ipfs_hash = ipfs_storage.store_data(data, encryption_key)
print(f"Stored data at IPFS hash: {ipfs_hash}")
retrieved_data = ipfs_storage.retrieve_data(ipfs_hash, encryption_key)
print(f"Retrieved data: {retrieved_data}")
