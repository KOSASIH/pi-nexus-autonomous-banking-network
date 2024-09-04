import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class PaymentGateway:
    def __init__(self, private_key_file: str, public_key_file: str, payment_processor_url: str):
        self.private_key_file = private_key_file
        self.public_key_file = public_key_file
        self.payment_processor_url = payment_processor_url
        self.private_key = self.load_private_key()
        self.public_key = self.load_public_key()

    def load_private_key(self) -> rsa.RSAPrivateKey:
        with open(self.private_key_file, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

    def load_public_key(self) -> rsa.RSAPublicKey:
        with open(self.public_key_file, "rb") as f:
            return serialization.load_pem_public_key(f.read(), backend=default_backend())

    def generate_payment_token(self, payment_info: dict) -> str:
        # Generate a payment token using the private key
        token = self.private_key.sign(
            json.dumps(payment_info).encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return token.hex()

    def send_payment_request(self, payment_token: str) -> dict:
        # Send the payment token to the payment processor
        response = requests.post(self.payment_processor_url, json={"payment_token": payment_token})
        return response.json()

# Example usage
gateway = PaymentGateway("private_key.pem", "public_key.pem", "https://payment-processor.com/process")
payment_info = {"amount": 10.99, "currency": "USD", "device_id": "device1"}
payment_token = gateway.generate_payment_token(payment_info)
response = gateway.send_payment_request(payment_token)
print(response)  # Output: {"payment_status": "success", "transaction_id": "1234567890"}
