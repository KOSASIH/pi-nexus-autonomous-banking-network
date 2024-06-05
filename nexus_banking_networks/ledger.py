class NexusLedger:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction: NexusTransaction):
        self.transactions.append(transaction)

    # Other methods for querying and managing the ledger
