from models.transaction import Transaction
from utils.security import encrypt, decrypt

class TransactionService:
    def __init__(self, config: dict):
        self.config = config

    def create_transaction(self, transaction: Transaction) -> Transaction:
        # Transaction creation logic here
        encrypted_transaction = encrypt(transaction, self.config["encryption_key"])
        return encrypted_transaction

    def get_transaction(self, transaction_id: int) -> Optional[Transaction]:
        # Database query logic here
        transaction_data =...  # retrieve transaction data from database
        decrypted_transaction = decrypt(transaction_data, self.config["decryption_key"])
        return decrypted_transaction
