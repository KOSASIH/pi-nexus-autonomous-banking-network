import hashlib
import json
from time import time

class Transaction:
    def __init__(self, sender, recipient, amount, timestamp=None):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.timestamp = timestamp or time()
        self.signature = None
        self.multi_signatures = []

    def calculate_hash(self):
        transaction_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(transaction_string).hexdigest()

    def sign_transaction(self, private_key):
        # Sign the transaction with the sender's private key
        self.signature = self._create_signature(private_key)

    def _create_signature(self, private_key):
        # Implement signature creation logic (e.g., using ECDSA)
        return "signature_based_on_private_key"

    def add_multi_signature(self, signature):
        self.multi_signatures.append(signature)

    def validate_transaction(self):
        # Validate the transaction's signature
        if not self.signature:
            return False
        
        # Here you would verify the signature using the sender's public key
        return self._verify_signature(self.sender, self.signature)

    def _verify_signature(self, public_key, signature):
        # Implement signature verification logic
        return True  # Placeholder for actual verification logic

    def is_fungible(self):
        # Check if the transaction is fungible (e.g., currency transfer)
        return isinstance(self.amount, (int, float)) and self.amount > 0

    def to_dict(self):
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'multi_signatures': self.multi_signatures
        }

class SmartContractTransaction(Transaction):
    def __init__(self, sender, recipient, amount, contract_code, timestamp=None):
        super().__init__(sender, recipient, amount, timestamp)
        self.contract_code = contract_code  # Smart contract code

    def execute_contract(self):
        # Execute the smart contract logic
        # This is a placeholder for actual contract execution logic
        return f"Executing contract: {self.contract_code}"

    def validate_contract(self):
        # Validate the smart contract code (e.g., syntax check)
        return True  # Placeholder for actual validation logic
