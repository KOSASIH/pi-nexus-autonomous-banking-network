import os
import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class SecurityOracle:
    def __init__(self, private_key_path, public_key_path):
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        self.private_key = self.load_private_key()
        self.public_key = self.load_public_key()

    def load_private_key(self):
        with open(self.private_key_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )
        return private_key

    def load_public_key(self):
        with open(self.public_key_path, 'rb') as key_file:
            public_key = serialization.load_ssh_public_key(
                key_file.read(),
                backend=default_backend()
            )
        return public_key

    def generate_keys(self):
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        private_key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_key_pem = key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )
        with open(self.private_key_path, 'wb') as private_key_file:
            private_key_file.write(private_key_pem)
        with open(self.public_key_path, 'wb') as public_key_file:
            public_key_file.write(public_key_pem)

    def sign_data(self, data):
        signature = self.private_key.sign(
            data.encode('utf-8'),
            padding=serialization.pkcs7.PKCS7Options.Default(),
            algorithm=serialization.rsassa.PSS(
                mgf=serialization.rsassa.MGF1(algorithm=hashes.SHA256()),
                salt_length=serialization.rsassa.PSS.MAX_LENGTH
            )
        )
        return signature

    def verify_signature(self, data, signature):
        try:
            self.public_key.verify(
                signature,
                data.encode('utf-8'),
                padding=serialization.pkcs7.PKCS7Options.Default(),
                algorithm=serialization.rsassa.PSS(
                    mgf=serialization.rsassa.MGF1(algorithm=hashes.SHA256()),
                    salt_length=serialization.rsassa.PSS.MAX_LENGTH
                )
            )
            return True
        except InvalidSignature:
            return False

class SecurityOracleAPI:
    def __init__(self, oracle):
        self.oracle = oracle

    def generate_keys(self):
        self.oracle.generate_keys()
        return {'message': 'Keys generated successfully'}

    def sign_data(self, data):
        signature = self.oracle.sign_data(data)
        return {'signature': signature.hex()}

    def verify_signature(self, data, signature):
        signature_bytes = bytes.fromhex(signature)
        if self.oracle.verify_signature(data, signature_bytes):
            return {'message': 'Signature verified successfully'}
        else:
            return {'message': 'Invalid signature'}, 401

if __name__ == '__main__':
    private_key_path = 'private_key.pem'
    public_key_path = 'public_key.pub'
    oracle = SecurityOracle(private_key_path, public_key_path)
    api = SecurityOracleAPI(oracle)
    app = Flask(__name__)
    api.add_resource(api.generate_keys, '/generate_keys')
    api.add_resource(api.sign_data, '/sign_data')
    api.add_resource(api.verify_signature, '/verify_signature')
    app.run(debug=True)
