import os
import time
from threading import Thread
from agents.bank_integration import get_bank_balance, create_bank_transaction
from services.monitoring import monitor_network_performance
from services.analytics import analyze_network_traffic

# Load private key from environment variable or generate a new one
private_key = os.getenv('PRIVATE_KEY') or generate_private_key()

# Set the network accelerator's performance threshold
PERFORMANCE_THRESHOLD = 50

# Set the network accelerator's monitoring interval
MONITORING_INTERVAL = 10

def accelerate_bank_transactions():
    while True:
        # Monitor network performance
        network_performance = monitor_network_performance()

        # Check if network performance is below the threshold
        if network_performance < PERFORMANCE_THRESHOLD:
            # Get the balance of all banks in the network
            bank_balances = {bank_id: get_bank_balance(bank_id) for bank_id in get_all_bank_ids()}

            # Identify banks with the highest and lowest balances
            highest_balance_bank_id, lowest_balance_bank_id = find_banks_with_highest_lowest_balances(bank_balances)

            # Create a transaction between the banks with the highest and lowest balances
            if highest_balance_bank_id and lowest_balance_bank_id:
                amount = calculate_transaction_amount(highest_balance_bank_id, lowest_balance_bank_id, bank_balances)
                create_bank_transaction(highest_balance_bank_id, lowest_balance_bank_id, amount)

        # Analyze network traffic
        analyze_network_traffic()

        # Sleep for the monitoring interval
        time.sleep(MONITORING_INTERVAL)

def find_banks_with_highest_lowest_balances(bank_balances):
    # Sort the bank balances in ascending and descending order
    sorted_bank_balances = sorted(bank_balances.items(), key=lambda x: x[1])

    # Return the bank IDs with the highest and lowest balances
    return sorted_bank_balances[-1][0], sorted_bank_balances[0][0]

def calculate_transaction_amount(highest_balance_bank_id, lowest_balance_bank_id, bank_balances):
    # Calculate the transaction amount based on the balances of the two banks
    highest_balance = bank_balances[highest_balance_bank_id]
    lowest_balance = bank_balances[lowest_balance_bank_id]

    # Calculate the transaction amount as a percentage of the highest balance
    transaction_percentage = 0.01
    transaction_amount = highest_balance * transaction_percentage

    # Ensure the transaction amount is within the acceptable range
    if transaction_amount < 100:
        transaction_amount = 100
    elif transaction_amount > 10000:
        transaction_amount = 10000

    return transaction_amount

# Start the network accelerator thread
accelerator_thread = Thread(target=accelerate_bank_transactions)
accelerator_thread.start()
