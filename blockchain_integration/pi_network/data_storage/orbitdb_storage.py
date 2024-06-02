import orbitdb
import json
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class OrbitDBStorage:
    def __init__(self, orbitdb_url, orbitdb_port, private_key_path):
        self.orbitdb_client = orbitdb.Client(f"ws://{orbitdb_url}:{orbitdb_port}")
        self.db = self.orbitdb_client.db("mydatabase")
        self.private_key = self._load_private_key(private_key_path)
        self.data_cache = {}

    async def store_data(self, data, encryption_key):
        # Encrypt data using the provided encryption key
        encrypted_data = self._encrypt_data(data, encryption_key)

        # Add data to OrbitDB
        doc_id = await self.db.put(encrypted_data)

        # Store doc ID in cache
        self.data_cache[doc_id] = encrypted_data

        return doc_id

    async def retrieve_data(self, doc_id, encryption_key):
        # Check if data is cached
        if doc_id in self.data_cache:
            encrypted_data = self.data_cache[doc_id]
        else:
            # Retrieve data from OrbitDB
            encrypted_data = await self.db.get(doc_id)

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
orbitdb_storage = OrbitDBStorage("localhost", 9000, "path/to/private/key")
data = {"hello": "world"}
encryption_key = "my_secret_key"
doc_id = await orbitdb_storage.store_data(data, encryption_key)
print(f"Stored data at OrbitDB document ID: {doc_id}")
retrieved_data = await orbitdb_storage.retrieve_data(doc_id, encryption_key)
print(f"Retrieved data: {retrieved_data}")
