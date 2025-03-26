class Bank:

    def __init__(self, bank_id, name, attributes):
        self.bank_id = bank_id
        self.name = name
        self.attributes = attributes
        self.accounts = {}

    def add_account(self, account_id, account_type, attributes):
        account = Account(account_id, account_type, attributes)
        self.accounts[account_id] = account

    def get_accounts(self):
        return list(self.accounts.values())

    def get_account(self, account_id):
        return self.accounts.get(account_id)
