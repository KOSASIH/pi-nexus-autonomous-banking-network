# eonix_transaction.py
import hashlib
import json
from eonix_token import EonixToken

class EonixTransaction:
    def __init__(self, sender, recipient, amount, token: EonixToken, transaction_fee=0):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.token = token
        self.transaction_fee = transaction_fee
        self.transaction_id = self.generate_transaction_id()

    def generate_transaction_id(self):
        transaction_id_hash = hashlib.sha256(f"{self.sender}{self.recipient}{self.amount}{self.token.get_token_id()}{self.transaction_fee}".encode()).hexdigest()
        return transaction_id_hash[:32]  # 32-character transaction ID

    def get_sender(self):
        return self.sender

    def get_recipient(self):
        return self.recipient

    def get_amount(self):
        return self.amount

    def get_token(self):
        return self.token

    def get_transaction_fee(self):
        return self.transaction_fee

    def get_transaction_id(self):
        return self.transaction_id

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "token": self.token.to_dict(),
            "transaction_fee": self.transaction_fee,
            "transaction_id": self.transaction_id
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, transaction_dict):
        token_dict = transaction_dict["token"]
        token = EonixToken.from_dict(token_dict)
        return cls(
            transaction_dict["sender"],
            transaction_dict["recipient"],
            transaction_dict["amount"],
            token,
            transaction_dict["transaction_fee"]
        )

    @classmethod
    def from_json(cls, transaction_json):
        transaction_dict = json.loads(transaction_json)
        return cls.from_dict(transaction_dict)

    def validate(self):
        # Implement transaction validation logic here
        # For example:
        if self.amount <= 0:
            raise ValueError("Transaction amount must be greater than 0")
        if self.transaction_fee < 0:
            raise ValueError("Transaction fee must be non-negative")
        # Add more validation rules as needed
        return True
