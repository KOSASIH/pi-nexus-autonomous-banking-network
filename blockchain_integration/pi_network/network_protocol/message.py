class Message:
    def __init__(self, sender, receiver, data):
        self.sender = sender
        self.receiver = receiver
        self.data = data

    def __repr__(self):
        return f"Message({self.sender}, {self.receiver}, {self.data})"
