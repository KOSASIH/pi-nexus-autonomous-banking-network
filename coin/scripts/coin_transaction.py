from services.transaction_service import TransactionService
from models.transaction import Transaction

def create_transaction(coin_id: int, amount: float) -> Transaction:
    config =...  # load config from config.json
    transaction_service = TransactionService(config)
    transaction = Transaction(coin_id=coin_id, amount=amount)
    created_transaction = transaction_service.create_transaction(transaction)
    return created_transaction
