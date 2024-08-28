import pandas as pd

def load_transaction_data(file_path):
    """
    Load transaction data from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Transaction data
    """
    transaction_data = pd.read_csv(file_path)
    return transaction_data

def load_market_data(file_path):
    """
    Load market data from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Market data
    """
    market_data = pd.read_csv(file_path)
    return market_data

def load_network_data(file_path):
    """
    Load network data from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Network data
    """
    network_data = pd.read_csv(file_path)
    return network_data

def load_user_behavior_data(file_path):
    """
    Load user behavior data from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: User behavior data
    """
    user_behavior_data = pd.read_csv(file_path)
    return user_behavior_data

def preprocess_data(data):
    """
    Preprocess data by handling missing values and converting data types.

    Args:
        data (pd.DataFrame): Data to be preprocessed

    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Convert data types
    data['date'] = pd.to_datetime(data['date'])

    return data
