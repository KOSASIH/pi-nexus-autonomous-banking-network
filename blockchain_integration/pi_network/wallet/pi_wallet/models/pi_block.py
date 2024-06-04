class PiBlock:

    def __init__(self, block_number, transactions):
        self.block_number = block_number
        self.transactions = transactions
        self.timestamp = int(time.time())

    def to_dict(self):
        return {
            "block_number": self.block_number,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
        }
