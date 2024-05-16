from flask_security import Security, SQLAlchemyUserDatastore, User, Role, RoleUser
from config import encrypt_data, decrypt_data

# ...

class Nexus(Flask):
    # ...

    def __init__(self):
        # ...

        self.user_datastore = SQLAlchemyUserDatastore(self.db, User, Role)
        self.security = Security(self, self.user_datastore)

        # Encrypt account data
        self.accounts = {
            account_id: {
                "balance": encrypt_data(str(balance)),
                "transactions": encrypt_data(json.dumps(transactions))
            }
            for account_id, (balance, transactions) in accounts.items()
        }

    def get_account_balance(self, account_id):
        account = self.accounts.get(account_id)
        if account:
            return float(decrypt_data(account["balance"]))
        return None

    def get_account_transactions(self, account_id):
        account = self.accounts.get(account_id)
        if account:
            return json.loads(decrypt_data(account["transactions"]))
        return None
