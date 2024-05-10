from datetime import datetime
from banking.accounts import BankAccount

def transfer(from_account: BankAccount, to_account: BankAccount, amount):
    """
    Transfers the specified amount from the from_account to the to_account.
    """
    from_account.withdraw(amount)
    to_account.deposit(amount)

    transaction_date = datetime.now()
    from_account_number = from_account.account_number
    to_account_number = to_account.account_number
    transaction_amount = amount

    print(f"Transfer from account {from_account_number} to account {to_account_number} for ${transaction_amount:.2f} on {transaction_date}")
