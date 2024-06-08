class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.unconfirmed_transactions = []

    def create_genesis_block(self):
        return Block(0, "0", [], int(time.time()))

    def add_block(self, block):
        self.chain.append(block)

    def get_latest_block(self):
        return self.chain[-1]

    def get_unconfirmed_transactions(self):
        return self.unconfirmed_transactions

    def add_transaction(self, transaction):
        self.unconfirmed_transactions.append(transaction)
