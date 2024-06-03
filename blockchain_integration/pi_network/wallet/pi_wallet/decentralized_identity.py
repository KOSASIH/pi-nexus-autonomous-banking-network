import uport
from uport.did import DID
from uport.credentials import Credentials
from uport.utils import encode_data, decode_data

class DecentralizedIdentity:
    def __init__(self, did, private_key):
        self.did = DID(did)
        self.private_key = private_key
        self.uport = uport.Uport(self.did, self.private_key)
        self.credentials = Credentials(self.did, self.private_key)

    def create_identity(self):
        self.uport.create_identity()

    def get_identity(self):
        return self.uport.get_identity()

    def update_identity(self, updated_identity):
        self.uport.update_identity(updated_identity)

    def add_credential(self, credential):
        self.credentials.add_credential(credential)

    def get_credentials(self):
        return self.credentials.get_credentials()

    def verify_credential(self, credential):
        return self.credentials.verify_credential(credential)

    def sign_data(self, data):
        return self.uport.sign_data(data)

    def verify_signature(self, data, signature):
        return self.uport.verify_signature(data, signature)

    def encrypt_data(self, data):
        return self.uport.encrypt_data(data)

    def decrypt_data(self, encrypted_data):
        return self.uport.decrypt_data(encrypted_data)

# Example usage:
did = "did:uport:0x..."
private_key = "0x..."
decentralized_identity = DecentralizedIdentity(did, private_key)

# Create identity
decentralized_identity.create_identity()

# Get identity
identity = decentralized_identity.get_identity()
print(identity)

# Update identity
updated_identity = {"name": "John Doe", "email": "johndoe@example.com"}
decentralized_identity.update_identity(updated_identity)

# Add credential
credential = {"type": "UniversityDegree", "issuer": "University of Blockchain", "issued": "2022-01-01"}
decentralized_identity.add_credential(credential)

# Getcredentials
credentials = decentralized_identity.get_credentials()
print(credentials)

# Verify credential
verified = decentralized_identity.verify_credential(credential)
print(verified)

# Sign data
data = "Hello, world!"
signature = decentralized_identity.sign_data(data)
print(signature)

# Verify signature
verified = decentralized_identity.verify_signature(data, signature)
print(verified)

# Encrypt data
encrypted_data = decentralized_identity.encrypt_data(data)
print(encrypted_data)

# Decrypt data
decrypted_data = decentralized_identity.decrypt_data(encrypted_data)
print(decrypted_data)
