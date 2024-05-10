from banking.accounts import BankAccount
from banking.transactions import transfer


def create_account(account_number):
    """
    Creates a new bank account with the specified account number.
    """
    return BankAccount(account_number)


def get_account(account_number):
    """
    Retrieves the bank account with the specified account number.
    """
    # Load account data from database or other storage system
    account_data = load_account_data(account_number)

    # Create new account if it doesn't exist
    if account_data is None:
        account = create_account(account_number)
    else:
        account = BankAccount(account_number, balance=account_data["balance"])

    return account


def save_account(account):
    """
    Saves the specified bank account to the database or other storage system.
    """
    # Save account data to database or other storage system
    save_account_data(account.account_number, account.balance)


def process_transaction(transaction):
    """
    Processes the specified transaction by transferring funds between bank accounts.
    """
    from_account = get_account(transaction["from_account"])
    to_account = get_account(transaction["to_account"])
    amount = transaction["amount"]

    transfer(from_account, to_account, amount)

    save_account(from_account)
    save_account(to_account)
