# utils/data_processing.py
def process_data(data):
    # ...

# transaction_processing/tasks.py
from utils.data_processing import process_data

def process_transaction(transaction_data):
    process_data(transaction_data)
    # ...

# account_management/tasks.py
from utils.data_processing import process_data

def process_account_update(account_data):
    process_data(account_data)
    # ...
