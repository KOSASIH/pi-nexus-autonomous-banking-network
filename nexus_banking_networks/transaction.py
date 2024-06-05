class NexusTransaction:

    def __init__(self, sender: NexusAccount, receiver: NexusAccount, amount: float):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

    # Other methods for processing and validating transactions
