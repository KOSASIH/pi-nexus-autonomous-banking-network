class Account:
    def __init__(self, account_id, account_type, attributes):
        self.account_id = account_id
        self.account_type = account_type
        self.attributes = attributes
        self.balance = 0.0

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient balance")
        self.balance -= amount

    def get_balance(self):
        return self.balance
