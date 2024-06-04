class PiAccount:

    def __init__(self, address):
        self.address = address
        self.private_key = None
        self.public_key = None
        self.balance = 0

    def set_private_key(self, private_key):
        self.private_key = private_key

    def set_public_key(self, public_key):
        self.public_key = public_key

    def update_balance(self, amount):
        self.balance += amount

    def get_balance(self):
        return self.balance
