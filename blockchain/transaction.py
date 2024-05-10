class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

    def to_string(self):
        return "Sender: " + self.sender + ", Receiver: " + self.receiver + ", Amount: " + str(self.amount)
