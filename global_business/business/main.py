import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from business import Business

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    business = Business()

    # Create a new customer
    customer_id = business.create_customer(name="John Doe", email="john.doe@example.com", address="123 Main St", phone_number="123-456-7890")
    logger.info(f"Created customer {customer_id}")

    # Create a new account for the customer
    account_number = business.create_account(customer_id=customer_id, account_type="checking")
    logger.info(f"Created account {account_number} for customer {customer_id}")

    # Get the account and customer
    account = business.get_account(account_number)
    customer = business.get_customer(customer_id)

    # Deposit money into the account
    account.deposit(1000.0)
    logger.info(f"Deposited $1000.00 into account {account_number}")

    # Withdraw money from the account
    account.withdraw(200.0)
    logger.info(f"Withdrew $200.00 from account {account_number}")

    # Process a transaction
    business.process_transaction(account_number, "deposit", 500.0)
    logger.info(f"Processed deposit of $500.00 for account {account_number}")

if __name__ == "__main__":
    main()
