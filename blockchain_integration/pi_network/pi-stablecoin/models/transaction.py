import hashlib

class Transaction:
  def __init__(self, sender, recipient, amount):
    self.sender = sender
    self.recipient = recipient
    self.amount = amount
    self.hash = self.calculate_hash()

  def calculate_hash(self):
    data_string = str(self.sender) + str(self.recipient) + str(self.amount)
    return hashlib.sha256(data_string.encode()).hexdigest()
