import logging


class Business:
    def __init__(self):
        self.customers: Dict[str, Customer] = {}
        self.accounts: Dict[str, Account] = {}

    def create_customer(
        self, name: str, email: str, address: str, phone_number: str
    ) -> str:
        """Create a new customer and return the customer ID.

        Args:
            name (str): The name of the customer.
            email (str): The email address of the customer.
            address (str): The address of the customer.
            phone_number (str): The phone number of the customer.

        Returns:
            str: The customer ID.
        """
        customer_id = str(uuid4())[:8]
        self.customers[customer_id] = Customer(
            customer_id, name, email, address, phone_number
        )
        return customer_id

    def create_account(self, customer_id: str, account_type: str) -> str:
        """Create a new account for a customer and return the account number.

        Args:
            customer_id (str): The customer ID.
            account_type (str): The type of account to create.

        Returns:
            str: The account number.
        """
        if customer_id not in self.customers:
            raise ValueError("Customer not found")

        account = Account(account_type=account_type, balance=0.0)
        self.accounts[account.account_number] = account

        self.customers[customer_id].accounts.append(account.account_number)
        return account.account_number

    def get_account(self, account_number: str) -> Optional[Account]:
        """Get an account by account number.

        Args:
            account_number (str): The account number.

        Returns:
            Optional[Account]: The account, or None if not found.
        """
        return self.accounts.get(account_number)

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        """Get a customer by customer ID.

        Args:
            customer_id (str): The customer ID.

        Returns:
            Optional[Customer]: The customer, or None if not found.
        """
        return self.customers.get(customer_id)

    def process_transaction(
        self, account_number: str, transaction_type: str, amount: float
    ):
        """Process a transaction for an account.

        Args:
            account_number (str): The account number.
            transaction_type (str): The type of transaction.
            amount (float): The amount of the transaction.
        """
        account = self.get_account(account_number)
        if account is None:
            raise ValueError("Account not found")

        transaction = Transaction(account, transaction_type, amount)
        transaction.process()
