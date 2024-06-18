from stellar_sdk import Transaction, Asset, Memo

class StellarTransaction:
    def __init__(self, source_account, destination_account, amount, asset_code, asset_issuer):
        self.source_account = source_account
        self.destination_account = destination_account
        self.amount = amount
        self.asset_code = asset_code
        self.asset_issuer = asset_issuer

    def build_transaction(self):
        transaction = Transaction(
            source=self.source_account,
            sequence=self.get_sequence_number(),
            fee=100,
            operations=[
                Payment(
                    destination=self.destination_account,
                    asset=Asset(self.asset_code, self.asset_issuer),
                    amount=self.amount
                )
            ],
            memo=Memo(text='Test transaction')
        )
        return transaction

    def get_sequence_number(self):
        # implement sequence number retrieval logic
        pass
