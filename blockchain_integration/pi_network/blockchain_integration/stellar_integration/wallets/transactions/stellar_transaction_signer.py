from stellar_sdk import TransactionEnvelope

class StellarTransactionSigner:
    def __init__(self, wallet):
        self.wallet = wallet

    def sign_transaction(self, transaction):
        envelope = TransactionEnvelope(transaction, network_passphrase="Test SDF Network ; September 2015")
        signed_envelope = self.wallet.sign_transaction(envelope)
        return signed_envelope
