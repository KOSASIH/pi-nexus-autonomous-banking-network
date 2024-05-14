import secrets

def create_account(name: str) -> Account:
    account_number = secrets.token_hex(16)
    account = Account(name, account_number)
    return account
