class BankAccount:
    """
    Represents a bank account with a balance and transaction history.
    """

    def __init__(self, account_number, balance=0):
        """
        Initializes a new bank account with the specified account number and balance.
        """
        self.account_number = account_number
        self.balance = balance
        self.transaction_history = []

    def deposit(self, amount):
        """
        Deposits the specified amount into the bank account.
        """
        self.balance += amount
        self.transaction_history.append(f"Deposit: ${amount:.2f}")

    def withdraw(self, amount):
        """
        Withdraws the specified amount from the bank account, if sufficient funds are available.
        """
        if self.balance >= amount:
            self.balance -= amount
            self.transaction_history.append(f"Withdraw: ${amount:.2f}")
        else:
            raise InsufficientFundsError(
                f"Insufficient funds to withdraw ${amount:.2f} from account {self.account_number}"
            )


class InsufficientFundsError(Exception):
    """
    Raised when there are insufficient funds to complete a withdrawal.
    """

    pass
