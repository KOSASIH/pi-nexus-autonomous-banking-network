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
    try:
        logging.info(f"Depositing {amount} into account {account_id}")
        account = get_account_by_id(account_id)
        if amount < 0:
            raise ValueError("Amount cannot be negative")
        account['balance'] += amount
        save_account(account)
        return True
    except Exception as e:
        logging.error(f"Error depositing into account {account_id}: {e}")
        return False

def withdraw(account_id: int, amount: float) -> bool:
    """
    Withdraw an amount from an account.

    Args:
        account_id (int): The ID of the account.
        amount (float): The amount to withdraw.

    Returns:
        bool: True if the withdrawal was successful, False otherwise.
    """
    try:
        logging.info(f"Withdrawing {amount} from account {account_id}")
        account = get_account_by_id(account_id)
        if amount < 0:
            raise ValueError("Amount cannot be negative")
        if account['balance'] < amount:
            raise ValueError("Insufficient balance")
        account['balance'] -= amount
        save_account(account)
        return True
    except Exception as e:
        logging.error(f"Error withdrawing from account {account_id}: {e}")
        return False

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
    try:
        logging.info(f"Transferring {amount} from account {from_account_id} to account {to_account_id}")
        if not withdraw(from_account_id, amount):
            return False
        if not deposit(to_account_id, amount):
            # Rollback the withdrawal if deposit fails
            deposit(from_account_id, amount)
            return False
        return True
    except Exception as e:
        logging.error(f"Error transferring between accounts: {e}")
        return False
