class Transaction:

    def __init__(self, transaction_id, account_id, amount, attributes):
        self.transaction_id = transaction_id
        self.account_id = account_id
        self.amount = amount
        self.attributes = attributes

    def execute(self, account):
        if self.amount > 0:
            account.deposit(self.amount)
        else:
            account.withdraw(-self.amount)
