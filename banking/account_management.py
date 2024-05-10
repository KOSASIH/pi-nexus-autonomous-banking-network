# banking/account_management.py

import logging
from typing import Dict

def create_account(account_type: str, balance: float, customer_id: int) -> Dict:
    """
    Create a new account for a customer.

    Args:
        account_type (str): The account type (e.g., 'checking', 'savings').
        balance (float): The initial account balance.
        customer_id (int): The ID of the customer.

    Returns:
        Dict: A dictionary containing the account information.
    """
    # implementation
    logging.info(f"Creating account for customer {customer_id}")
    # Validate input parameters
    if account_type not in ['checking', 'savings']:
        raise ValueError("Invalid account type")
    if balance < 0:
        raise ValueError("Balance cannot be negative")

    # Create account and store it in a database or in-memory data structure
    account = {
        'id': generate_unique_id(),
        'type': account_type,
        'balance': balance,
        'customer_id': customer_id
    }

    # Return the account information
    return account

def update_account(account_id: int, account_type: str, balance: float) -> bool:
    """
    Update an existing account.

    Args:
        account_id (int): The ID of the account.
        account_type (str): The updated account type.
        balance (float): The updated account balance.

    Returns:
        bool: True if the account was updated, False otherwise.
    """
    # implementation
    logging.info(f"Updating account {account_id}")
    # Validate input parameters
    if account_type not in ['checking', 'savings']:
        raise ValueError("Invalid account type")
    if balance < 0:
        raise ValueError("Balance cannot be negative")

    # Retrieve the account from the database or in-memory data structure
    account = get_account_by_id(account_id)

    # Update the account information
    account['type'] = account_type
    account['balance'] = balance

    # Save the updated account back to the database or in-memory data structure
    save_account(account)

    # Return True to indicate success
    return True

def delete_account(account_id: int) -> bool:
    """
    Delete an account.

    Args:
        account_id (int): The ID of the account.

    Returns:
        bool: True if the account was deleted, False otherwise.
    """
    # implementation
    logging.info(f"Deleting account {account_id}")
    # Retrieve the account from the database or in-memory data structure
    account = get_account_by_id(account_id)

    # Delete the account
    delete_account_from_database(account_id)

    # Return True to indicate success
    return True
