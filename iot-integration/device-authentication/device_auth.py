import base64
import hashlib
import hmac
import json
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class DeviceAuth:

    def __init__(self, device_id, device_secret, auth_token_expiration=3600):
        self.device_id = device_id
        self.device_secret = device_secret
        self.auth_token_expiration = auth_token_expiration
        self.private_key = self.generate_private_key()
        self.public_key = self.generate_public_key()

    def generate_private_key(self):
        """Generate a private key for device authentication"""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        return private_key

    def generate_public_key(self):
        """Generate a public key for device authentication"""
        public_key = self.private_key.public_key()
        return public_key

    def generate_auth_token(self):
        """Generate an authentication token for the device"""
        payload = {"device_id": self.device_id, "exp": self.auth_token_expiration}
        header = {"typ": "JWT", "alg": "RS256"}
        header_json = json.dumps(header, separators=(",", ":"))
        payload_json = json.dumps(payload, separators=(",", ":"))
        header_base64 = base64.urlsafe_b64encode(header_json.encode()).decode()
        payload_base64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
        signature = self.sign(header_base64, payload_base64)
        auth_token = f"{header_base64}.{payload_base64}.{signature}"
        return auth_token

    def sign(self, header_base64, payload_base64):
        """Sign the authentication token using the private key"""
        message = f"{header_base64}.{payload_base64}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        signature_base64 = base64.urlsafe_b64encode(signature).decode()
        return signature_base64

    def verify_auth_token(self, auth_token):
        """Verify the authentication token using the public key"""
        header_base64, payload_base64, signature_base64 = auth_token.split(".")
        message = f"{header_base64}.{payload_base64}".encode()
        signature = base64.urlsafe_b64decode(signature_base64.encode())
        try:
            self.public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def get_device_id(self):
        """Get the device ID"""
        return self.device_id

    def get_device_secret(self):
        """Get the device secret"""
        return self.device_secret

    def get_auth_token_expiration(self):
        """Get the authentication token expiration time"""
        return self.auth_token_expiration
