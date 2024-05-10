# banking/transaction_processing.py

import logging


def deposit(account_id: int, amount: float) -> bool:
    """
    Deposit an amount into an account.

    Args:
        account_id (int): The ID of the account.
        amount (float): The amount to deposit.

    Returns:
        bool: True if the deposit was successful, False otherwise.
    """
    # implementation
    logging.info(f"Depositing {amount} into account {account_id}")
    # Retrieve the account from the database or in-memory data structure
    account = get_account_by_id(account_id)

    # Validate the input parameters
    if amount < 0:
        raise ValueError("Amount cannot be negative")

    # Update the account balance
    account["balance"] += amount

    # Save the updated account back to the database or in-memory data structure
    save_account(account)

    # Return True to indicate success
    return True


def withdraw(account_id: int, amount: float) -> bool:
    """
    Withdraw an amount from an account.

    Args:
        account_id (int): The ID of the account.
        amount (float): The amount to withdraw.

    Returns:
        bool: True if the withdrawal was successful, False otherwise.
    """
    # implementation
    logging.info(f"Withdrawing {amount} from account {account_id}")
    # Retrieve the account from the database or in-memory data structure
    account = get_account_by_id(account_id)

    # Validate the input parameters
    if amount < 0:
        raise ValueError("Amount cannot be negative")
    if account["balance"] < amount:
        raise ValueError("Insufficient balance")

    # Update the account balance
    account["balance"] -= amount

    # Save the updated account back to the database or in-memory data structure
    save_account(account)

    # Return True to indicate success
    return True


def transfer(from_account_id: int, to_account_id: int, amount: float) -> bool:
    """
    Transfer an amount between two accounts.

    Args:
        from_account_id (int): The ID of the source account.
        to_account_id (int): The ID of the destination account.
        amount (float): The amount to transfer.

    Returns:
        bool: True if the transfer was successful, False otherwise.
    """
    # implementation
    logging.info(
        f"Transferring {amount} from account {from_account_id} to account {to_account_id}"
    )
    # Withdraw the amount from the source account
    withdraw(from_account_id, amount)

    # Deposit the amount into the destination account
    deposit(to_account_id, amount)

    # Return True to indicate success
    return True
