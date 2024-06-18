from stellar_sdk import Server

class StellarHorizon:
    def __init__(self, horizon_url):
        self.server = Server(horizon_url)

    def get_account(self, account_id):
        return self.server.accounts().account_id(account_id).call()

    def get_transaction(self, transaction_id):
        return self.server.transactions().transaction_id(transaction_id).call()

    def submit_transaction(self, transaction):
        return self.server.submit_transaction(transaction)
