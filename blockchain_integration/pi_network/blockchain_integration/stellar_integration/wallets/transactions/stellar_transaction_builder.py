from stellar_sdk import TransactionBuilder

class StellarTransactionBuilder:
    def __init__(self, source_account, destination_account, amount, asset_code, asset_issuer):
        self.source_account = source_account
        self.destination_account = destination_account
        self.amount = amount
        self.asset_code = asset_code
        self.asset_issuer = asset_issuer

    def build_transaction(self):
        transaction_builder = TransactionBuilder(
            source_account=self.source_account,
            network_passphrase="Test SDF Network ; September 2015",
            base_fee=100
        )
        transaction_builder.add_operation(
            Payment(
                destination=self.destination_account,
                asset=Asset(self.asset_code, self.asset_issuer),
                amount=self.amount
            )
        )
        return transaction_builder
