import hashlib
import json


class Transaction:
    def __init__(self, sender, recipient, amount, fee=1):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.fee = fee
        self.timestamp = None
        self.hash = None

    def calculate_hash(self):
        # Calculate the hash of the transaction
        transaction_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(transaction_string).hexdigest()

    def validate(self):
        # Validate the transaction
        if self.sender == self.recipient:
            return False
        if self.amount <= 0:
            return False
        if self.fee <= 0:
            return False
        if not isinstance(self.sender, str):
            return False
        if not isinstance(self.recipient, str):
            return False
        if not isinstance(self.amount, int):
            return False
        if not isinstance(self.fee, int):
            return False
        return True

    def to_json(self):
        # Convert the transaction to a JSON string
        transaction_dict = self.__dict__
        transaction_dict["hash"] = self.hash
        return json.dumps(transaction_dict)
