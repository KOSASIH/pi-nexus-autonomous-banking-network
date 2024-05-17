def create_account(account_number: int, initial_balance: float) -> dict:
    """
    Creates a new bank account with the given account number and initial balance.

    Args:
        account_number (int): The account number.
        initial_balance (float): The initial balance.

    Returns:
        dict: A dictionary representing the new account.
    """
    try:
        # Create a new account
        account = {"account_number": account_number, "balance": initial_balance}
        # Add account to database or storage
        # ...
        return account
    except Exception as e:
        print(f"Error creating account: {e}")
        return None
