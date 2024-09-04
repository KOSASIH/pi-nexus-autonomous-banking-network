import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class PaymentProcessor:
    def __init__(self, public_key_file: str):
        self.public_key_file = public_key_file
        self.public_key = self.load_public_key()

    def load_public_key(self) -> rsa.RSAPublicKey:
        with open(self.public_key_file, "rb") as f:
            return serialization.load_pem_public_key(f.read(), backend=default_backend())

    def process_payment(self, payment_token: str) -> dict:
        # Verify the payment token using the public key
        try:
            payment_info = self.public_key.verify(
                bytes.fromhex(payment_token),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            payment_info = json.loads(payment_info.decode())
            # Process the payment
            # ...
            return {"payment_status": "success", "transaction_id": "1234567890"}
        except Exception as e:
            return {"payment_status": "failed", "error": str(e)}

# Example usage
processor = PaymentProcessor("public_key.pem")
payment_token = "1234567890abcdef"
response = processor.process_payment(payment_token)
print(response)  # Output: {"payment_status": "success", "transaction_id": "1234567890"}
