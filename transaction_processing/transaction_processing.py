import time

class TransactionProcessor:
    def __init__(self, transaction_validator, transaction_router):
        self.transaction_validator = transaction_validator
        self.transaction_router = transaction_router

    def process_transaction(self, transaction):
        errors = self.transaction_validator.validate_transaction(transaction)
        if errors:
            print('Transaction errors:', errors)
            return False
        else:
            channel = self.transaction_router.route_transaction(transaction)
            print('Transaction routed to:', channel)
            time.sleep(1)
            print('Transaction processed successfully.')
            return True
