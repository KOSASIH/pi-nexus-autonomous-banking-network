from .stellar_service import StellarService

class StellarTransactionManager:
    def __init__(self, stellar_service):
        self.stellar_service = stellar_service

    def create_transaction(self, source, destination, asset, amount):
        transaction = self.stellar_service.create_transaction(source, destination, asset, amount)
        return transaction

    def sign_and_submit_transaction(self, transaction, keypair):
        signed_transaction = self.stellar_service.sign_transaction(transaction, keypair)
        response = self.stellar_service.submit_transaction(signed_transaction)
        return response
