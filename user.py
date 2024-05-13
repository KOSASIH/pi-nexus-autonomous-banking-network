class User:
    def __init__(self, name, account_number):
        self.name = name
        self.account_number = account_number
        self.balance = 0

    def deposit(self, amount):
        """
        Deposit an amount of money into the user's account.

        :param amount: The amount of money to deposit.
        :return: None
        """
        if amount > 0:
            self.balance += amount
            print(f"{amount} deposited successfully. Current balance: {self.balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        """
        Withdraw an amount of money from the user's account.

        :param amount: The amount of money to withdraw.
        :return: None
        """
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            print(f"{amount} withdrawn successfully. Current balance: {self.balance}")
        else:
            print("Invalid withdrawal amount.")

    def get_balance(self):
        """
        Get the user's account balance.

        :return: The account balance.
        """
        return self.balance
